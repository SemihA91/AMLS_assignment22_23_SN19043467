from A1 import gender_detection as a1
import os

def create_datasets_dir():   
    if not os.path.exists('Datasets/'):
        os.makedirs('Datasets')
    return

def main():
    create_datasets_dir()
    a1.run_classifier()
    

if __name__ == "__main__":
    main()
