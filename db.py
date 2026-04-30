import psycopg2

from dotenv import dotenv_values

config = dotenv_values('.env')

DB_HOST = config['DB_HOST']
DB_USER = config['DB_USER']
DB_PW = config["DB_PW"]
DB_NAME = config["DB_NAME"]
DB_PORT = config["DB_PORT"]

try:
    conn = psycopg2.connect(
        host = DB_HOST,
        dbname = DB_NAME,
        password = DB_PW,
        user = DB_USER,
        port = DB_PORT
    )

    # print(conn)
    cur = conn.cursor()

    query = 'SELECT area, price FROM house;'
    cur.execute(query)

    rows = cur.fetchall() # varg

    for row in rows:
        print(row)

except Exception as e:
    print("Error: ", e)
finally:
    if 'cur' in locals():
        cur.close()
    if 'conn' in locals():
        conn.close()
    print("Connection is closed")