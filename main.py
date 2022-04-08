import datetime
from distilling import model_train
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--model', type=str, default='teacher')

    args = parser.parse_args()

    tb_path = './tensorboard/{}/'.format(args.model)
    model_path = './cheakpoint/{}/'.format(args.model)

    if args.model == 'teacher' or args.model == 'student':
        model_train.clf_train(model_name=args.model, tb_path=tb_path, model_path=model_path)

    if args.model == 'distilling':
        model_train.distilling_train(model_name=args.model, tb_path=tb_path, model_path=model_path)

# teacher 0.8938
# student 0.8718

