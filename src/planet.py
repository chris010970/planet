import os
import glob
import math
import random
import pandas as pd
import numpy as np

from skimage.io import imread
from sklearn.utils import shuffle as pd_shuffle
from sklearn.preprocessing import MultiLabelBinarizer

class Planet:

    def __init__( self ):

        """ 
        constructor
        """

        # dataset vital statistics
        self._width = 256; self._height = 256
        self._channels = 3

        # zip class names into dict
        labels = [  'agriculture', 
                    'artisinal_mine', 
                    'bare_ground', 
                    'blooming', 
                    'blow_down', 
                    'clear', 
                    'cloudy', 
                    'conventional_mine', 
                    'cultivation', 
                    'habitation', 
                    'haze', 
                    'partly_cloudy', 
                    'primary', 
                    'road', 
                    'selective_logging', 
                    'slash_burn', 
                    'water' ]

        # assign label to unique id
        self._classes = dict(zip( labels,range (len(labels) ) ) )
        return


    def getSubsetFiles( self, data_path ):

        """ 
        subset original training data into train and test 
        """

        # load file as csv and shuffle
        df = pd.read_csv( os.path.join( data_path, 'train_v2.csv' ) )
        df = pd_shuffle( df )

        split_index = int( len( df ) * 0.8 )

        # assign 80% records as training
        train = df [ 0: split_index ]
        train.to_csv( os.path.join( data_path, 'train.csv' ), index=False )
        
        # assign remaining 20% records as validation / test
        test = df [ split_index : len( df ) ]
        test.to_csv( os.path.join( data_path, 'test.csv' ), index=False )
        
        return


    def getClassSampleSizes( self, pathname ):

        """ 
        get class sample sizes
        """

        # load file as csv and split tags into tuples
        df = pd.read_csv( pathname )
        df[ 'tags' ] = df.tags.apply(lambda x: tuple( x.split(' ') ) )

        # initialise sample count array
        sample_count = np.zeros( len( self._classes ) )
        for idx, row in df.iterrows():

            # increment counter linked to class id
            for item in row[ 'tags' ]:
                sample_count[ self._classes[ item ] ] += 1

        return sample_count


    def getNormalisationStats( self, data_path ):

        """
        compute mean and variance of images 
        """

        # initialise stats
        sum_x = np.zeros( self._channels ); sum_x2 = np.zeros( self._channels )
        count = np.zeros( self._channels )

        # separately process each class sub-directory
        files = glob.glob( os.path.join( data_path, 'train/*.jpg' ) )
        for f in files:

            # load image
            image = np.array( imread( f ), dtype=float )
            for channel in range( self._channels ):

                # flatten channel data
                data = np.reshape(image[:,:,channel], -1)
                count[ channel ] += data.shape[ 0 ]

                # update sum and sum of squares 
                sum_x[ channel ] += np.sum( data )
                sum_x2[ channel ] += np.sum( data**2 )


        # for each channel
        stats = []
        for channel in range( self._channels ):

            # compute mean and stdev from summations
            mean = sum_x[ channel ] / count [ channel ]
            stdev = math.sqrt ( sum_x2[ channel ] / count [ channel ] - mean**2 )

            # append channel stats to list
            stats.append ( [ channel, mean, stdev ] )

        # convert list to dataframe and save to csv
        df = pd.DataFrame( stats, columns =['channel', 'mean', 'stdev'], dtype = float ) 
        df.to_csv( os.path.join( data_path, 'stats.csv' ), index=False )

        return


    def updateDataFrame( self, df, path ):

        """ 
        tweak data frame content before initiating training
        add full image pathname and transform tag lists into one hot vector
        """

        # convert tags to tuple - add full image pathname
        df[ 'pathname' ] = df.image_name.apply( lambda x: os.path.join( path, x + '.jpg' ) )
        df[ 'tags' ] = df.tags.apply(lambda x: tuple( x.split(' ') ) )

        # create and fit multilabel binariser
        mlb = MultiLabelBinarizer( list ( self._classes.keys() ) )
        df[ 'target' ] = tuple( mlb.fit_transform( df['tags'] ) )

        return df


# get sample sizes + files
# obj = Planet()
# data_path = 'C:\\Users\\Chris.Williams\\Documents\\GitHub\\planet\\data\\'

# obj.getSubsetFiles( data_path )
# print ( obj.getClassSampleSizes( os.path.join( data_path, 'train_v2.csv' ) ) )
# print ( obj.getClassSampleSizes( os.path.join( data_path, 'train.csv' ) ) )
# obj.getNormalisationStats( data_path )
