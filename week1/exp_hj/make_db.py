import duckdb
import pandas as pd

'''

'''

# csv to df
df = pd.read_csv('movie_list.csv')

# DuckDB
con = duckdb.connect(database='week1.duckdb')
con.execute("CREATE TABLE movies AS SELECT * FROM df")

result = con.execute('SELECT * FROM movies').fetchall()

con.close()
