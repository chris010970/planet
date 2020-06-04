import os
import argparse
import numpy as np
import pandas as pd

# callbacks
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# keras
from keras.optimizers import SGD
#from tensorflow.keras.metrics import AUC, Precision, Recall

# local imports
from fbeta import fbeta
from model import getVgg16, getResNet50
from model import saveToFile, loadFromFile
from planet import Planet
from generator import MultiChannelImageDataGenerator


class Train:

    def __init__( self, args ):

        """ 
        constructor
        """

        # append topmost layers to architecture
        self._planet = Planet()

        # get input shape and top most layer definitions
        in_shape = ( self._planet._height, self._planet._width, self._planet._channels )
        top_layers = {  'fc' : [    { 'units' : 256, 'activation' : 'relu', 'dropout' : 0.2 },
                                    { 'units' : 128, 'activation' : 'relu', 'dropout' : 0.2 } ],
                        'out' : [   { 'units' : len( self._planet._classes ), 'activation' : 'sigmoid' } ]
        }

        # create custom vgg16 architecture
        if args.architecture == 'vgg16':
            self._model = getVgg16( in_shape, top_layers, weights='imagenet' )

        # create custom resnet50 architecture
        if args.architecture == 'resnet50':
            self._model = getResNet50( in_shape, top_layers )

        self._model.summary()
        return


    def process( self, args ):

        """ 
        main path of execution
        """

        # load model weights from file
        if args.load_path is not None:
            self._model, args.architecture = loadFromFile( args.load_path )

        # get stats dataframe
        # use pretrained imagenet weights - normalise around mean 
        # stats = pd.read_csv( os.path.join( args.data_path, 'stats.csv' ) )
        stats = pd.DataFrame ( { 'mean':  [123.68, 116.779, 103.939] }, columns=['mean'] )

        # get train and test dataframes
        df = {  'train' : pd.read_csv( os.path.join( args.data_path, 'train.csv' ) ),
                'test' : pd.read_csv( os.path.join( args.data_path, 'test.csv' ) ) }

        # stratify into class-ordered groups to address sampling imbalance
        groups = { 'train' : [], 'test' : [] }
        for subset in [ 'train', 'test' ]:

            # update content in core dataframes
            df[ subset ] = self._planet.updateDataFrame(    df[ subset ], 
                                                            os.path.join( args.data_path, 'train' ) )

            # split core frame into stratified groups
            for label in self._planet._classes.keys():
                groups[ subset ].append( df[ subset ] [ df[ subset ][ 'tags' ].apply( lambda x: label in x ) ] )
                
        # get train generator
        train_generator = MultiChannelImageDataGenerator(   groups[ 'train' ],
                                                            args.batch_size,
                                                            stats=stats,
                                                            horizontal_flip=args.horizontal_flip,
                                                            vertical_flip=args.vertical_flip,
                                                            rotation_range=args.rotation,
                                                            shear_range=args.shear,
                                                            scale_range=args.scale,
                                                            transform_range=args.transform,
                                                            filling_mode='edge',
                                                            speckle=args.speckle,
                                                            crop=args.crop,
                                                            crop_size=args.crop_size )

        # get test generator
        test_generator = MultiChannelImageDataGenerator(    groups[ 'test' ],
                                                            args.batch_size,
                                                            stats=stats )

        # setup optimiser and compile
        opt = SGD( lr=0.0001, momentum=0.9 )
        self._model.compile(    optimizer=opt, 
                                loss='binary_crossentropy', 
                                metrics=[fbeta] )
                                #metrics=[AUC(name='auc'),Precision(name='precision'),Recall(name='recall') ] )

        # setup callbacks
        callbacks = [ CSVLogger( 'log.csv', append=True ) ]
        if args.checkpoint_path is not None:

            # create sub-directory if required
            if not os.path.exists ( args.checkpoint_path ):
                os.makedirs( args.checkpoint_path )

            # setup checkpointing callback
            pathname = os.path.join( args.checkpoint_path, "weights-{epoch:02d}-{val_fbeta:.2f}.h5" )
            checkpoint = ModelCheckpoint(   pathname, 
                                            monitor='val_fbeta', 
                                            verbose=1, 
                                            save_best_only=True, 
                                            mode='max' )
            callbacks.append( checkpoint )

        # initiate training
        self._model.fit_generator(  train_generator,
                                    steps_per_epoch=args.train_steps,
                                    epochs=args.epochs,
                                    callbacks=callbacks,
                                    validation_data=test_generator,
                                    validation_steps=args.validation_steps )

        history = self._model.history

        # save final set of weights to file
        if args.save_path is not None:
            saveToFile( self._model, args.save_path, args.architecture )

        return


def parseArguments(args=None):

    """
    parse command line arguments
    """

    # parse configuration
    parser = argparse.ArgumentParser(description='planet train')
    parser.add_argument('data_path', action='store')

    parser.add_argument(    '--architecture', 
                            action='store',
                            default='vgg16' )

    # epochs
    parser.add_argument(    '--epochs', 
                            type=int,
                            action='store',
                            default=50 )

    # steps per epoch
    parser.add_argument(    '--train_steps', 
                            type=int,
                            action='store',
                            default=300 )

    parser.add_argument(    '--validation_steps', 
                            type=int,
                            action='store',
                            default=50 )

    # batch size
    parser.add_argument(    '--batch_size', 
                            type=int,
                            action='store',
                            default=16 )

    # -------------- augmentation parameters -----------------------

    # affine transform
    parser.add_argument(    '--rotation', 
                            type=int,
                            action='store',
                            default=0 )

    parser.add_argument(    '--shear', 
                            type=int,
                            action='store',
                            default=0 )

    parser.add_argument(    '--scale', 
                            type=int,
                            action='store',
                            default=1 )

    parser.add_argument(    '--transform', 
                            type=int,
                            action='store',
                            default=0 )

    # flip parameters
    parser.add_argument(    '--horizontal_flip', 
                            type=bool,
                            action='store',
                            default=False )

    parser.add_argument(    '--vertical_flip', 
                            type=bool,
                            action='store',
                            default=False )

    # centre crop parameters
    parser.add_argument(    '--crop', 
                            type=bool,
                            action='store',
                            default=False )

    parser.add_argument(    '--crop_size', 
                            type=int,
                            action='store',
                            default=None )

    # speckle noise
    parser.add_argument(    '--speckle', 
                            type=float,
                            action='store',
                            default=None )

    # warp fill mode
    parser.add_argument(    '--filling_mode', 
                            action='store',
                            default='edge' )

    # checkpoint path
    parser.add_argument(    '--checkpoint_path', 
                            action='store',
                            default=None )

    # save model path
    parser.add_argument(    '--save_path', 
                            action='store',
                            default=None )

    # load model path
    parser.add_argument(    '--load_path', 
                            action='store',
                            default=None )


    return parser.parse_args(args)


def main():

    """
    main path of execution
    """

    # parse arguments
    args = parseArguments()
    
    # create and execute training instance
    obj = Train( args )
    obj.process( args )

    return


# execute main
if __name__ == '__main__':
    main()

