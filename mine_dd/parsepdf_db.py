import datetime
import os
import sqlite3
from getpass import getpass

import nest_asyncio
import pandas as pd
from llama_parse import LlamaParse

nest_asyncio.apply()

class DatabaseHandler:
    """SQL database handling; connection, table creation, and data inserts."""
    def __init__(self, db_file):
        self.db_file = db_file
        self.conn = self.create_connection()

    def create_connection(self):
        """Create a database connection to a SQLite database."""
        return sqlite3.connect(self.db_file)

    def create_db_tables(self):
        """Create tables literature_pages and literature_fulltext in given database."""
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

        cursor = self.conn.cursor()
        cursor.execute(querypages)
        cursor.execute(queryfulltext)
        self.conn.commit()

        return

    def close_connection(self):
        """Close to connection to the SQL database."""
        return self.conn.close()

    def insert_pages(self, md, data_pages):
        """Create SQL insert query for individual paper pages.

        This function:
        - Sets up a SQL query template to insert values in a table.
        - Organises the data for a bulk import to the sql database.
        """
        query = """
            INSERT INTO literature_pages (filename, title, authors, DOI, publicationyear,
                journal, pages, page, fulltext)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ;"""

        queryvalues = []
        for i in range(len(data_pages)):
            queryvalues.append((md['PDF Name'], md['Name'], md['Authors'], md['DOI'], md['Year'],
                                md['Journal'], len(data_pages), i, data_pages[i].text))

        cursor = self.conn.cursor()
        cursor.executemany(query, queryvalues)
        self.conn.commit()

        return

    def insert_fulltext(self, metadata, data_fulltext):
        """Create SQL insert query template for full text papers."""
        query = """
            INSERT INTO literature_fulltext (filename, title, authors, DOI, publicationyear,
                journal, fulltext)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ;"""

        queryvalues = (metadata['PDF Name'], metadata['Name'], metadata['Authors'], metadata['DOI'],
                    metadata['Year'], metadata['Journal'], data_fulltext[0].text)

        # Execute query
        cursor = self.conn.cursor()
        cursor.execute(query, queryvalues)
        self.conn.commit()

        return query


class PdfParser:
    """LlaMa file parser."""
    def __init__(self, llama_key):
        os.environ["LLAMA_CLOUD_API_KEY"] = llama_key

    def parse_files(self, file, page=True):
        """Parse file usingn LlamaPase."""
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

        return parser.load_data(file)


class MetaData:
    """Metadata handler."""
    def __init__(self, metadata_file):
        self.df = pd.read_csv(metadata_file)

    def get_metadata(self, file):
        """Select metadata for given filename."""
        filename = os.path.basename(file)
        metadata = self.df[self.df['PDF Name'] == filename]

        try:
            return metadata[['PDF Name', 'Name', 'Authors', 'DOI', 'Year', 'Journal'
                             ]].to_dict(orient='records')[0]
        except IndexError:
            errordata = open("errordata.txt", "a")
            errordata.write(f"{datetime.datetime.now()}, failed: {filename} \n")
            errordata.close()
            return


class PdfParsedDatbase:
    """Parse pdf"""
    def __init__(self, db_file, paperfolder, metadata_file):
        self.database_handler = DatabaseHandler(db_file)
        self.paperfolder = paperfolder
        self.metadata = MetaData(metadata_file)

    def parse_pdfs_to_db(self):
        """Parse pdf files to sql database.

        This function takes a pdf file and parses it with LLaMa, both as fulltext and text per page.
        Together with the corresponding metadata of the document, the data is stored into a created
        sql database, in two different tables; one for page data and one for fulltext, the file
        metadata is added to both tables.
        """
        self.database_handler.create_db_tables()

        llama_key = getpass("LlaMa cloud API key: ")
        parser = PdfParser(llama_key)

        with os.scandir(self.paperfolder) as entries:
            full_file_paths = [os.path.join(self.paperfolder, entry.name) for entry in entries
                               if entry.is_file() and entry.name.endswith(".pdf")]

        for i, file in enumerate(full_file_paths):
            print(f"Parsing file {i} out of {len(full_file_paths)}, {os.path.basename(file)}")
            metadata = self.metadata.get_metadata(file)

            if metadata:
                data_pages = parser.parse_files(file, page=True)
                self.database_handler.insert_pages(metadata, data_pages)

                data_fulltext = parser.parse_files(file, page=False)
                self.database_handler.insert_fulltext(metadata, data_fulltext)

        # Close db connection
        self.database_handler.close_connection()

        return


if __name__ == '__main__':
    database = "literature.db"
    paperfolder = input("Path of the folder with pdf papers: ")
    metadata_file = input("Path of the csv file with metadata: ")

    PdfParsedDatbase(database, paperfolder, metadata_file).parse_pdfs_to_db()
