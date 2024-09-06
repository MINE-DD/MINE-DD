import os
from llama_parse import LlamaParse
import nest_asyncio
import sqlite3
from getpass import getpass

nest_asyncio.apply()

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = sqlite3.connect(db_file)
    return conn

def insert_query_pages(data, table):
    """Return SQL insert query"""
    query = """
        INSERT INTO literature_pages (title, page, publicationdate, pagetext)
            values ("filetitle.pdf", 0, "01-01-2000", "the bodytext of the paper")
        ;"""
    return query


def insert_query_fulltext(data, table):
    """Return SQL insert query"""
    query = """
        INSERT INTO literature_pages (title, page, publicationdate, fulltext)
            values ("filetitle.pdf", 0, "01-01-2000", "the bodytext of the paper")
        ;"""
    return query

def create_database(db):
    
    querypages = """CREATE TABLE IF NOT EXISTS literature_pages
    (title TEXT,
    page INT,
    publicationdate DATE,
    text TEXT
    );"""

    queryfulltext = """CREATE TABLE IF NOT EXISTS literature_fulltext
    (title TEXT,
    publicationdate DATE,
    text TEXT
    );"""

    conn = create_connection(db)
    cursor = conn.cursor()
    cursor.execute(querypages)
    cursor.execute(queryfulltext)
    conn.commit()
    conn.close


def fill_database(db, filename):
    "return a page databse and a full database"
    # Create database connection
    conn = create_connection(db)
    cursor = conn.cursor()

    # pages
    data_pages = parse_files(files, page=True)
    query = insert_query_pages(data_pages, "literature_pages")

    # full text
    cursor.exexcute(query)
    conn.commit()
    # Close database connection
    conn.close()


def parse_files(file, page: bool=True):
    parser = LlamaParse(
        result_type="markdown",  # "markdown" and "text" are available
        num_workers=8,
        verbose=True,
        language="en", 
        show_progress=True,
        fast_mode=False,
        parsing_instruction= """ 
            This is a research paper. 
            """,
        split_by_page=page
    )
    parsed_doc = parser.load_data(file)
    return parsed_doc


if __name__ == '__main__':
    llama_key = getpass("LlaMa cloud API key: ")
    os.environ["LLAMA_CLOUD_API_KEY"] = llama_key
    database = "literature.db"
    create_database(database)

    folder = "../notebooks/data"
    with os.scandir(folder) as entries:
        full_file_paths = [os.path.join(folder, entry.name) for entry in entries if entry.is_file()]

    for i, file in enumerate(full_file_paths):
        print(f"Parsing file {i} out of {len(full_file_paths)}, {file}")
        fill_database(database, file)
