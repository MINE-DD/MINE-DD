import os
from llama_parse import LlamaParse
import nest_asyncio
import sqlite3
from getpass import getpass
import pandas as pd
import datetime

nest_asyncio.apply()

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = sqlite3.connect(db_file)
    return conn

def insert_query_pages(md, data_pages, table):
    """Return SQL insert query"""
    queryvalues = []
    for i in range(len(data_pages)):
        queryvalues.append((md['PDF Name'], md['Name'], md['Authors'], md['DOI'], md['Year'], md['Journal'], len(data_pages), i, data_pages[i].text))

    query = f"""
        INSERT INTO {table} (filename, title, authors, DOI, publicationyear, journal, pages, page, fulltext)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ;"""
    return query, queryvalues


def insert_query_fulltext(table):
    """Return SQL insert query"""

    query = f"""
        INSERT INTO {table} (filename, title, authors, DOI, publicationyear, journal, fulltext)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ;"""
    return query


def create_database(db):
    
    querypages = """
    CREATE TABLE IF NOT EXISTS literature_pages
    (   filename TEXT,
        title TEXT,
        authors TEXT,
        DOI TEXT,
        publicationyear DATE,
        journal TEXT,
        pages INT,
        page INT,
        fulltext TEXT
    );"""

    queryfulltext = """
    CREATE TABLE IF NOT EXISTS literature_fulltext
    (   filename TEXT,
        title TEXT,
        authors TEXT,
        publicationyear DATE,
        DOI TEXT,
        journal TEXT,
        fulltext TEXT
    );"""

    conn = create_connection(db)
    cursor = conn.cursor()
    cursor.execute(querypages)
    cursor.execute(queryfulltext)
    conn.commit()
    conn.close

    return


def fill_database(db, filename, metadata):
    "return a page database and a full database"
    # Create database connection
    conn = create_connection(db)
    cursor = conn.cursor()

    # pages
    data_pages = parse_files(file, page=True)
    querypages, qvp = insert_query_pages(filename, metadata, data_pages, "literature_pages")
    cursor.executemany(querypages, qvp)

    # full text
    data_fulltext = parse_files(file, page=False)
    queryfull = insert_query_fulltext(filename, metadata, data_fulltext, "literature_fulltext")
    queryvalues = (metadata['PDF Name'], metadata['Name'], metadata['Authors'], metadata['DOI'],
                   metadata['Year'], metadata['Journal'], data_fulltext[0].text)
    cursor.execute(queryfull, queryvalues)

    # execute query
    conn.commit()
    conn.close()


def parse_files(file, page: bool=True):
    parser = LlamaParse(
        result_type="markdown",
        num_workers=8,
        verbose=True,
        language="en",
        skip_diagonal_text=True,
        show_progress=True,
        fast_mode=False,
        parsing_instruction= """
            This is a research paper.
            """,
        split_by_page=page
    )
    parsed_doc = parser.load_data(file)
    return parsed_doc


def get_metadata(df, filepath):
    filename = os.path.basename(filepath)
    metadata = df[df['PDF Name'] == filename]

    try:
        return metadata[['PDF Name', 'Name', 'Authors', 'DOI', 'Year', 'Journal']].to_dict(orient='records')[0]
    except IndexError:
        errordata = open("errordata.txt", "a")  
        errordata.write(f"{datetime.datetime.now()}, failed: {filepath} \n")
        errordata.close()
        return 


if __name__ == '__main__':
    llama_key = getpass("LlaMa cloud API key: ")
    os.environ["LLAMA_CLOUD_API_KEY"] = llama_key
    database = "literature.db"
    create_database(database)

    folder = "../../relevant"
    metadata_file = "../../relevant/result.csv"
    df_metadata = pd.read_csv(metadata_file)

    with os.scandir(folder) as entries:
        full_file_paths = [os.path.join(folder, entry.name) for entry in entries if entry.is_file() and entry.name.endswith(".pdf")]

    for i, file in enumerate(full_file_paths):
        print(f"Parsing file {i} out of {len(full_file_paths)}, {file}")
        metadata = get_metadata(df_metadata, file)
        if metadata: fill_database(database, file, metadata)
