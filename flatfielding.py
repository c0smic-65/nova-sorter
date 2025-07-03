import MySQLdb as mysql
import pandas as pd

from database import GetConnection

class FlatFieldModel():
    def __init__(self):
        """
        Initialise all variables to default values
        """
        # self.conn = GetConnection()
        self.conn = GetConnection(section='hess')
        sets = pd.read_sql("SELECT SetNum, Telescope, ValidFrom, Events "
                           "FROM HD_test.Calib3_FlatField_Set", self.conn).set_index('SetNum')
        
        self.df = sets[sets.Events>0].drop_duplicates(['ValidFrom', 'Telescope'], keep='last')
        self.df.ValidFrom = pd.to_datetime(self.df.ValidFrom)
        self.df = self.df.sort_values(by='ValidFrom')
        
    def get_FlatField(self, date, telId):
        tel_df = self.df[self.df.Telescope == telId]
        i = tel_df['ValidFrom'].searchsorted(date)
        setId = tel_df.index[i-1]

        return self.get_coeff(setId).set_index('Pixel')
    
    def get_coeff(self, setId):
        data = pd.read_sql("SELECT Pixel, Coefficient_hg, Coefficient_lg, charge_mean_hg, charge_mean_lg "
                           "FROM HD_test.Calib3_FlatField "
                           "WHERE SetNum = {}".format(setId), self.conn)
        return data