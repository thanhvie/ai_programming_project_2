import argparse

def get_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', type = str, default = 'flowers', 
                    help = 'path to the folder of data images')
    
    return parser.parse_args()