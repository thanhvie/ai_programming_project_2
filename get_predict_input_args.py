import argparse

def get_predict_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_path', type = str, default = 'flowers/valid/1/image_06739.jpg', 
                    help = 'path to the image')
    
    parser.add_argument('checkpoint', type = str, default = 'checkpoint.pth',
                    help = 'checkpoint path')
    
    return parser.parse_args()