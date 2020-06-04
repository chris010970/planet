import os
import time
import argparse
import numpy as np
import pandas as pd

# metrics
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix

# graphics
import seaborn as sn
import matplotlib.pyplot as plt

# local imports
from planet import Planet
from model import loadFromFile
from generator import MultiChannelImageDataGenerator


class Predict:

    def __init__( self, args ):

        """
        constructor
        """

        # initialise members            
        self._model, self._architecture = loadFromFile( args.model_path )
        self._planet = Planet()
        return


    def process( self, args ):

        """ 
        main path of execution
        """

        # get stats dataframe
        # stats = pd.read_csv( os.path.join( args.data_path, 'stats.csv' ) )
        stats = pd.DataFrame ( { 'mean':  [123.68, 116.779, 103.939] }, columns=['mean'] )

        # get train and test dataframes
        df = {  'train' : pd.read_csv( os.path.join( args.data_path, 'train.csv' ) ),
                'test' : pd.read_csv( os.path.join( args.data_path, 'test.csv' ) ) }

        # stratify into class-ordered groups to address sampling imbalance
        args.batch_size = 1
        for subset in [ 'test' ]:

            # update data frame
            df[ subset ] = self._planet.updateDataFrame(    df[ subset ], 
                                                            os.path.join( args.data_path, 'train' ) )

            # get true and predicted class labels
            actual = np.asarray( df[ subset ][ 'target' ].to_list() )
            predict = self.getPrediction( df[ subset ], stats, args )

            # compute and plot multilabel confusion matrices
            cms = self.getConfusionMatrices( actual, predict )
            self.plotConfusionMatrices( cms, sum(actual[ :, 0:17]), subset ) 

        return


    def getPrediction( self, df, stats, args ):
                
        """
        generate prediction for images referenced in data frame
        """

        # create generator
        generator = MultiChannelImageDataGenerator( [ df ],
                                                    args.batch_size,
                                                    stats=stats,
                                                    shuffle=False )
        # initiate prediction
        steps = len( df ) // args.batch_size
        y_pred = self._model.predict_generator( generator, steps=steps )

        # values above 0.5 indicate high confidence prediction
        return np.asarray( y_pred > 0.5, dtype=np.int32 )


    def getConfusionMatrices( self, actual, predict, labels=[ 'negative', 'positive' ] ):

        """
        compute confusion matrix for prediction
        """

        results = []

        # compute multilabel confusion matrices - 2 x 2 x classes 
        cms = multilabel_confusion_matrix( actual, predict)
        for cm in cms:

            # parse normalised confusion matrix into dataframe
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            results.append( pd.DataFrame( cm, index=labels, columns=labels ) )

        return results 


    def plotConfusionMatrices( self, cms, samples, subset ):

        """
        plot train and test confusion matrix
        """

        # create figure
        nrows=3; ncols=6

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 18))
        sn.set(font_scale=0.8) 

        # iterate through classes / confusion matrices
        col = 0; row = 0
        for idx, c in enumerate ( list ( self._planet._classes.keys() ) ):

            # plot heatmap - adjust font and label size
            sn.heatmap( cms[ idx ], annot=True, cbar=False, annot_kws={"size": 10}, fmt='.2f', ax=axes[ row ][ col ] )
            axes[ row ][ col ].set_title('{} - {} samples'.format ( c, samples[ idx ] ) )

            # move onto next row
            col += 1
            if col == ncols:
                row += 1
                col = 0

        # set figure title and plot
        fig.suptitle( 'Normalised Confusion Matrices: {}'.format( subset ) )
        plt.tight_layout()
        plt.show()

        return


def parseArguments(args=None):

    """
    parse command line arguments
    """

    # parse configuration
    parser = argparse.ArgumentParser(description='eurosat train')
    parser.add_argument('data_path', action='store')
    parser.add_argument('model_path', action='store')

    return parser.parse_args(args)


def main():

    """
    main path of execution
    """

    # parse arguments
    args = parseArguments()
    
    # create and execute training instance
    obj = Predict( args )
    obj.process( args )

    return


# execute main
if __name__ == '__main__':
    main()
