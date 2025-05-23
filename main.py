from src.moderate import moderate_content
import argparse

def main():
    parser = argparse.ArgumentParser(description="Content Moderation System")
    parser.add_argument('--text', type=str, help='Text to moderate')
    parser.add_argument('--file', type=str, help='File containing texts to moderate')
    
    args = parser.parse_args()
    
    if args.text:
        print(moderate_content(args.text))
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    print(f"'{line}': {moderate_content(line)}")
    else:
        print("Running in interactive mode (type 'exit' to quit)")
        while True:
            msg = input("Enter message (or 'exit' to quit): ")
            if msg.lower() == 'exit':
                break
            print(moderate_content(msg))

if __name__ == "__main__":
    main()
