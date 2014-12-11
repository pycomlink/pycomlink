#----------------------------------------------------------------------------
# Name:         io
# Purpose:      Input and output functions for commercial MW link (CML) data
#
# Authors:      Christian Chwala, Felix Keis
#
# Created:      01.12.2014
# Copyright:    (c) Christian Chwala 2014
# Licence:      The MIT License
#----------------------------------------------------------------------------


from datetime import datetime
import pandas as pd

import psycopg2
import psycopg2.extras
import sqlalchemy

from comlink import Comlink


def get_cml_data_from_IFU_database(cml_id, 
                                   t1_str, 
                                   t2_str,
                                   t_str_format='%Y-%m-%d',
                                   db_ip='172.27.60.177', 
                                   db_port='5432',
                                   db_user='MW_parser',
                                   db_password='*MW_parser',
                                   db_name='MW_link',
                                   db_schema_data='data',
                                   db_schema_info='info'):
    """Query CML data from a database
    
    Parameters
    ----------

    cml_id : str
        ID of the desired CML, e.g. 'MY4320_2_MY2291_4'.
    t1_str : str
        Start time of the desired date range of the data. The standard
        format is '%Y-%m-%d', e.g. '2014-10-30'. Different formats can
        be supplied as 't_str_format'.
    t2_str : str
        Same as 't1_str' but for the end of the date range
    t_str_format : str, optional
        Format string for 't1_str' and 't2_str' according to strptime
        format codes
    db_ip : str, optional
        IP address of database server.
    db_port : str, optional
        Database server port.
    db_user : str, optional
        Databse user.
    db_password : str, optional
        Database user password.
    db_name : str, optional
        Database name.
    db_schema: str, optional
        Schema name
    
    Returns
    -------

    cml : Comlink
        Comlink class object provided by pycomlink.comlink    
    
    """

    # Convert time strings to datetime    
    t1 = datetime.strptime(t1_str, t_str_format)
    t2 = datetime.strptime(t2_str, t_str_format)    
    
    # Connect to database
    db_connection = psycopg2.connect(database=db_name,
                                     user=db_user, 
                                     password=db_password, 
                                     host=db_ip, 
                                     port=db_port)

    # Check if table with CML ID exists
    if table_exists(db_connection, cml_id.lower(),db_schema_data):
        # Create SQL engine to be used by Pandas
        sql_engine = sqlalchemy.create_engine('postgresql://' + 
                                              db_user + 
                                              ':' + db_password +
                                              '@' + db_ip +
                                              ':' + db_port +
                                              '/' + db_name)
                
        # Query data from database using Pandas
        TXRX_df=pd.read_sql("""(SELECT * from """ + db_schema_data + 
                        """.""" + cml_id.lower() + 
                       """ WHERE TIMESTAMP >= '""" + str(t1) + 
                       """'::timestamp AND TIMESTAMP <= '"""
                       + str(t2) + """'::timestamp);""",sql_engine, 
                       index_col='timestamp')
        
        # Parse metadata
        # Get link information
        db_cursor = db_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        sql_query="""(SELECT * FROM """+db_schema_info+""".info_links WHERE link_id = %(id)s);"""
        db_cursor.execute(sql_query, {"id":cml_id})
        metadata_dict = db_cursor.fetchone()
        
        # Get information for both site ips
        for z in 'ab':
            sd=get_site_info(db_cursor,db_schema_info,'info_sites',metadata_dict['ip_'+z])
            metadata_dict['ort_'+z]=sd['ort']
            metadata_dict['lat_'+z]=sd['lat']
            metadata_dict['lon_'+z]=sd['lon']
            metadata_dict['vpsz_'+z]=sd['vpsz']
        
        # Build Comlink object from data and metadata
        cml = Comlink(metadata_dict, TXRX_df)
    else:
        ValueError('Table for MW_link_ID %s does not exists in database' % 
                   cml_id)
    db_connection.close()
    return cml
    

    
def get_site_info(db_cursor,schema_str,table_str,site_ip):
    """Get information for site IP
    Parameters:
        db_cursor: cursor of psycopg.connection
        schema_str,table_str: str
            Strings of schema.table with site information
        site_ip: str
            IP of Site
    """
    
    sql_query="""(SELECT * FROM """+schema_str+"""."""+table_str+""" WHERE ip = %(i)s);"""
    db_cursor.execute(sql_query, {"i":site_ip})
    site_dict = db_cursor.fetchone()
    return site_dict    
    
    
def table_exists(con, table_str,schema_str):
    """Check if a MW_link data table exists in the database
    
    Parameters
    ----------
    
    con : psycopg2.connection
        Database server connection object provided by psycopg2
    table_str : str
        String of table name for which the existenst will be checked
    schema_str : str
        String of schema name where table is located
        
    Returns
    -------
    
    exists : bool
        True or False, whether the table exists or not
        
    """
    
    exists = False
    try:
        cur = con.cursor()
        cur.execute("""select exists(select * from information_schema.tables where
                    table_name=%s and table_schema=%s)""", (table_str,schema_str))
        exists = cur.fetchone()[0]    
    except psycopg2.Error as e:
         print e  
    return exists