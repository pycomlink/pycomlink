# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 14:34:51 2014

@author: chwala-c
"""

# enable debugging
#import cgi, cgitb
#cgitb.enable()

from datetime import datetime
import psycopg2
import sqlalchemy

import pandas as pd

from comlink import Comlink

#tb=cgi.FieldStorage()
#tbb=tb["param1"].value

#from_time = pd.DatetimeIndex([dt])[0]

def get_cml_data_from_IFU_database(cml_id, 
                                   t1_str, 
                                   t2_str,
                                   t_str_format='%Y-%m-%d',
                                   db_ip='172.27.60.177', 
                                   db_port='5432',
                                   db_user='MW_parser',
                                   db_password='*MW_parser',
                                   db_name='MW_link'):
    '''
    Query CML data from a database
    '''
    
    t1 = datetime.strptime(t1_str, t_str_format)
    t2 = datetime.strptime(t2_str, t_str_format)
    
    
    db_connection = psycopg2.connect(database=db_name,
                                     user=db_user, 
                                     password=db_password, 
                                     host=db_ip, 
                                     port=db_port)

    if table_exists(db_connection, cml_id.lower()):
        sql_engine = sqlalchemy.create_engine('postgresql://' + 
                                              db_user + 
                                              ':' + db_password +
                                              '@' + db_ip +
                                              ':' + db_port +
                                              '/' + db_name)
                
        TXRX_df=pd.read_sql("""(SELECT * from """ + cml_id.lower() + 
                       """ WHERE TIMESTAMP >= '""" + str(t1) + 
                       """'::timestamp AND TIMESTAMP <= '"""
                       + str(t2) + """'::timestamp);""",sql_engine, 
                       index_col='timestamp')
        
        # TODO: Parse metadata
        metadata_dict = None
        cml = Comlink(metadata_dict, TXRX_df)
    else:
        ValueError('Table for MW_link_ID %s does not exists in database' % 
                   cml_id)
    db_connection.close()
    return cml
    
    
def table_exists(con, table_str):
    '''
    Check if a MW_link data table exists in the database
    '''
    
    exists = False
    try:
        cur = con.cursor()
        cur.execute("select exists(select relname from pg_class where relname='"
                    + table_str + "')")
        exists = cur.fetchone()[0]    
    except psycopg2.Error as e:
         print e  
    return exists