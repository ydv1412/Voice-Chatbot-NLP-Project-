# signore
from dialogue import handle_user_transcript
from session import get_session

if __name__ == "__main__":
    sid = "tester"           
    get_session(sid)        
    print("Text eval mode. Type queries (q to quit).")

    while True:
        user_in = input("You: ").strip()
        if not user_in or user_in.lower() in {"q", "quit", "exit"}:
            break
        reply = handle_user_transcript(user_in, session_id=sid)
        print("Bot:", reply)
