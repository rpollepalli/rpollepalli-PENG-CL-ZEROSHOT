from src.main.lab import classify

"""
This file will contain some sample code to send the output of the functions in lab.py to the 
console. You may modify this file in any way, it will not affect the test results.
"""
def main():
    user_input = input("insert a prompt here for classification: ")
    output = classify(user_input)
    print(output)

if __name__ == '__main__':
    main()
