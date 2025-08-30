#!/usr/bin/env python3
import json
import os
import sys
import time
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import threading
from queue import Queue
import ollama

# Configuration
MAX_RETRIES = 3
TIMEOUT_SECONDS = 60
MAX_CLARIFICATION_ROUNDS = 5
SESSION_DIR = "prompt_sessions"
DEFAULT_MODEL = "gemma3:4b"

class SessionManager:
    """Handles session persistence and loading"""
    
    @staticmethod
    def save_session(session_id: str, data: Dict):
        os.makedirs(SESSION_DIR, exist_ok=True)
        path = os.path.join(SESSION_DIR, f"{session_id}.json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_session(session_id: str) -> Optional[Dict]:
        path = os.path.join(SESSION_DIR, f"{session_id}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    @staticmethod
    def list_sessions() -> List[str]:
        return [f.replace('.json', '') for f in os.listdir(SESSION_DIR) if f.endswith('.json')]

class OllamaInterface:
    """Handles communication with local Ollama models"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        #self._verify_model()
    
    def _verify_model(self):
        """Check if the model is available in Ollama"""
        try:
            models = ollama.list()
            model_names = [m['name'] for m in models['models']]
            if self.model_name not in model_names:
                print(f"Warning: Model {self.model_name} not found. Available models: {', '.join(model_names)}")
                print(f"Pulling model {self.model_name}...")
                ollama.pull(self.model_name)
        except Exception as e:
            print(f"Error verifying model: {str(e)}")
            sys.exit(1)
    
    def _call_llm(self, messages: List[Dict], max_retries: int = MAX_RETRIES) -> str:
        """Internal LLM call with timeout and retries"""
        result_queue = Queue()
        
        def worker():
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=messages
                )
                result_queue.put(response['message']['content'])
            except Exception as e:
                result_queue.put(e)
        
        for attempt in range(max_retries):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(timeout=TIMEOUT_SECONDS)
            
            if not thread.is_alive():
                result = result_queue.get()
                if isinstance(result, Exception):
                    raise result
                return result
            
            print(f"LLM call timed out (attempt {attempt + 1}/{max_retries})")
        
        raise TimeoutError(f"LLM call stalled after maximum retries with model {self.model_name}")
    
    def generate_clarification(self, context: List[Dict]) -> str:
        """Generate next clarification question"""
        system_message = {
            "role": "system",
            "content": "You are an AI assistant that helps users refine their prompts. Based on the conversation, generate the next most important clarification question to improve the prompt. Respond with only the question text."
        }
        messages = [system_message] + context
        return self._call_llm(messages)
    
    def generate_refined_prompts(self, context: List[Dict]) -> Tuple[str, str]:
        """Generate final system and user prompts"""
        system_message = {
            "role": "system",
            "content": """You are an expert prompt engineer. Based on the conversation, create a refined system prompt and user prompt that incorporates all clarifications.
Format your response exactly as:
SYSTEM: [system prompt]
USER: [user prompt]"""
        }
        messages = [system_message] + context
        response = self._call_llm(messages)
        return self._parse_refined_prompts(response)
    
    def _parse_refined_prompts(self, response: str) -> Tuple[str, str]:
        """Parse LLM response into system and user prompts"""
        system_prompt = ""
        user_prompt = ""
        
        for line in response.split('\n'):
            if line.startswith("SYSTEM:"):
                system_prompt = line.replace("SYSTEM:", "").strip()
            elif line.startswith("USER:"):
                user_prompt = line.replace("USER:", "").strip()
        
        return system_prompt, user_prompt

class PromptRefiner:
    """Main application workflow"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.llm = OllamaInterface(model_name)
        self.current_session = None
    
    def start_new_session(self, initial_prompts: List[str]) -> str:
        """Initialize new refinement session"""
        session_id = f"session_{int(time.time())}"
        self.current_session = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "model_used": self.llm.model_name,
            "initial_prompts": initial_prompts,
            "refinements": []
        }
        return session_id
    
    def load_session(self, session_id: str) -> bool:
        """Load existing session"""
        session_data = SessionManager.load_session(session_id)
        if session_data:
            self.current_session = session_data
            # Update model to the one used in the session
            self.llm.model_name = session_data.get("model_used", DEFAULT_MODEL)
            return True
        return False
    
    def refine_prompt_interactive(self, prompt_index: int) -> Dict:
        """Refine a single prompt through clarification rounds (interactive)"""
        if not self.current_session:
            raise ValueError("No active session")
        
        initial_prompt = self.current_session["initial_prompts"][prompt_index]
        context = [{"role": "user", "content": initial_prompt}]
        refinement = {
            "initial_prompt": initial_prompt,
            "qa_pairs": [],
            "refined_system": "",
            "refined_user": ""
        }
        
        print(f"\nRefining prompt: {initial_prompt[:100]}...")
        
        for round_num in range(MAX_CLARIFICATION_ROUNDS):
            try:
                question = self.llm.generate_clarification(context)
                print(f"\nClarification Question {round_num + 1}: {question}")
                
                answer = input("Your answer (or 'skip' to end): ")
                if answer.lower() == 'skip':
                    break
                
                context.append({"role": "assistant", "content": question})
                context.append({"role": "user", "content": answer})
                refinement["qa_pairs"].append({"question": question, "answer": answer})
                
            except Exception as e:
                print(f"Error during clarification: {str(e)}")
                continue
        
        # Generate final refined prompts
        try:
            print("\nGenerating refined prompts...")
            system_prompt, user_prompt = self.llm.generate_refined_prompts(context)
            refinement["refined_system"] = system_prompt
            refinement["refined_user"] = user_prompt
            
            print("\n=== REFINED PROMPTS ===")
            print("SYSTEM PROMPT:")
            print(system_prompt)
            print("\nUSER PROMPT:")
            print(user_prompt)
            print("======================\n")
            
        except Exception as e:
            print(f"Error generating refined prompts: {str(e)}")
        
        self.current_session["refinements"].append(refinement)
        SessionManager.save_session(self.current_session["session_id"], self.current_session)
        return refinement
    
    def refine_prompt_non_interactive(self, prompt_index: int) -> Dict:
        """Refine a single prompt without user interaction"""
        if not self.current_session:
            raise ValueError("No active session")
        
        initial_prompt = self.current_session["initial_prompts"][prompt_index]
        context = [{"role": "user", "content": initial_prompt}]
        refinement = {
            "initial_prompt": initial_prompt,
            "qa_pairs": [],
            "refined_system": "",
            "refined_user": ""
        }
        
        print(f"\nRefining prompt: {initial_prompt[:100]}...")
        
        # Generate final refined prompts directly without clarifications
        try:
            print("Generating refined prompts...")
            system_prompt, user_prompt = self.llm.generate_refined_prompts(context)
            refinement["refined_system"] = system_prompt
            refinement["refined_user"] = user_prompt
            
            print("\n=== REFINED PROMPTS ===")
            print("SYSTEM PROMPT:")
            print(system_prompt)
            print("\nUSER PROMPT:")
            print(user_prompt)
            print("======================\n")
            
        except Exception as e:
            print(f"Error generating refined prompts: {str(e)}")
        
        self.current_session["refinements"].append(refinement)
        SessionManager.save_session(self.current_session["session_id"], self.current_session)
        return refinement
    
    def process_prompts_from_file(self, file_path: str) -> List[str]:
        """Load initial prompts from file"""
        with open(file_path, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts
    
    def list_sessions(self):
        """List all saved sessions"""
        sessions = SessionManager.list_sessions()
        if not sessions:
            print("No sessions found.")
            return
        
        print("\nAvailable sessions:")
        for session in sessions:
            print(f"- {session}")
    
    def display_session(self, session_id: str):
        """Display a specific session"""
        if not self.load_session(session_id):
            print(f"Session {session_id} not found.")
            return
        
        print(f"\nSession: {session_id}")
        print(f"Created: {self.current_session['created_at']}")
        print(f"Model: {self.current_session['model_used']}")
        print(f"Initial Prompts: {len(self.current_session['initial_prompts'])}")
        print(f"Refinements: {len(self.current_session['refinements'])}")
        
        for i, refinement in enumerate(self.current_session["refinements"]):
            print(f"\nRefinement {i+1}:")
            print(f"Initial: {refinement['initial_prompt'][:50]}...")
            print(f"System: {refinement['refined_system'][:50]}...")
            print(f"User: {refinement['refined_user'][:50]}...")

def run_interactive_mode(model_name: str):
    """Run the application in interactive mode"""
    refiner = PromptRefiner(model_name)
    
    while True:
        print("\nPrompt Refinement Tool")
        print("1. Start new session")
        print("2. Load existing session")
        print("3. List sessions")
        print("4. Exit")
        
        choice = input("Select option: ")
        
        if choice == '1':
            source = input("Enter prompt (or 'file' to load from file): ")
            
            if source.lower() == 'file':
                file_path = input("Enter file path: ")
                try:
                    initial_prompts = refiner.process_prompts_from_file(file_path)
                    print(f"Loaded {len(initial_prompts)} prompts from file")
                except Exception as e:
                    print(f"Error loading file: {str(e)}")
                    continue
            else:
                initial_prompts = [source]
            
            session_id = refiner.start_new_session(initial_prompts)
            print(f"Started session {session_id} with model {model_name}")
            
            for i, prompt in enumerate(initial_prompts):
                print(f"\nProcessing prompt {i+1}/{len(initial_prompts)}")
                refiner.refine_prompt_interactive(i)
        
        elif choice == '2':
            session_id = input("Enter session ID: ")
            refiner.display_session(session_id)
        
        elif choice == '3':
            refiner.list_sessions()
        
        elif choice == '4':
            break
        
        else:
            print("Invalid choice")

def run_non_interactive_mode(args):
    """Run the application in non-interactive mode"""
    refiner = PromptRefiner(args.model)
    
    # Handle list sessions
    if args.list:
        refiner.list_sessions()
        return
    
    # Handle display session
    if args.session:
        refiner.display_session(args.session)
        return
    
    # Handle prompt refinement
    if args.prompt or args.file:
        if args.prompt:
            initial_prompts = [args.prompt]
        elif args.file:
            try:
                initial_prompts = refiner.process_prompts_from_file(args.file)
                print(f"Loaded {len(initial_prompts)} prompts from file")
            except Exception as e:
                print(f"Error loading file: {str(e)}")
                return
        
        session_id = refiner.start_new_session(initial_prompts)
        print(f"Started session {session_id} with model {args.model}")
        
        for i, prompt in enumerate(initial_prompts):
            print(f"\nProcessing prompt {i+1}/{len(initial_prompts)}")
            refiner.refine_prompt_non_interactive(i)
        
        # Save to output file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(refiner.current_session, f, indent=2)
            print(f"Session saved to {args.output}")

def main():
    parser = argparse.ArgumentParser(description="Prompt Refinement Tool")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, 
                        help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--prompt", type=str, help="Single prompt to refine")
    parser.add_argument("--file", type=str, help="File containing prompts (one per line)")
    parser.add_argument("--session", type=str, help="Display a specific session")
    parser.add_argument("--list", action="store_true", help="List all saved sessions")
    parser.add_argument("--output", type=str, help="File to save the refined prompts")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # If no arguments are provided, run in interactive mode
    if len(sys.argv) == 1 or args.interactive:
        run_interactive_mode(args.model)
    else:
        run_non_interactive_mode(args)

if __name__ == "__main__":
    main()
