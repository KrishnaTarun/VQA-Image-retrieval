#File for arguments and options
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dialog', type=int, help='consider only Q&A', default=0)
    parser.add_argument(
        '--caption', type=int, help='consider caption only', default=0)
    parser.add_argument(
        '--combine', type=int, help='combine both dialog and caption', default=1)
    parser.add_argument(
        '--path_folder', type=str, help='path of folder containing data', default="data/VQA_IR_data")
    parser.add_argument(
        '--type', type=str, help='Easy or Hard', default="Easy")
    parser.add_argument(
        '--img_feat', help='folder to image features', default="data/img_feat")
    parser.add_argument(
        '--model_type', help='simple / vectorized / minibatch', default="vectorized")
    parser.add_argument(
        '--epochs', help='Number of epochs or training iterations', default=10)

    # Array for all arguments passed to script
    args = parser.parse_args()
    return args

