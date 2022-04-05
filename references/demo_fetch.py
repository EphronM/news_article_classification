from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import pandas as pd
from cassandra.query import SimpleStatement


def pandas_factory(colnames, rows):
    return pd.DataFrame(rows, columns=colnames)
    engine = sqlalchemy.create_engine('sql_server_connection string')

cloud_config= {
        'secure_connect_bundle': 'data\external\secure-connect-news-db.zip'
}
auth_provider = PlainTextAuthProvider('boEzrNbZRKkGwaBXvMkrZmuc', '6ZQ_Nk8S6-A7cOgxgiH0S8aNv4Ny0m.vMMJF4RjaKIPyXrNAF-1Avj,C9JFw,-WZ_A-JcQ21Q9Iy,rRgw,H.ndfMS-avDrhrvz9JoAI2zP7L9TbcwBYP98_cNlZd9wKF')
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect(wait_for_all_pools=True)


session.row_factory = pandas_factory
request_timeout = 60000
query = "SELECT * FROM news.movies"



statement = SimpleStatement(query, fetch_size=5000) 
rows = session.execute(statement)

df = rows._current_rows
print(df)