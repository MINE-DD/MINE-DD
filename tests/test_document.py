import pytest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import json

from minedd.document import DocumentPDF, DocumentMarkdown, process_from_grobid_chunks, get_documents_from_directory

@pytest.fixture
def mock_pdf_path():
    return "/fake/path/to/document.pdf"

@pytest.fixture
def mock_json_path():
    return "/fake/path/to/document.json"

@pytest.fixture
def mock_md_content():
    return "# Title\n\nThis is some markdown content.\n\n## Section\n\nMore content."

@pytest.fixture
def mock_grobid_json_per_section_content():
    return {
        "Intro": [
            {
            "text": "The global burden of diarrhoeal disease, the fourth leading cause of under-5 mortality, is likely to increase because of climate change.",
            "pages": "1-1",
            "chunk_index": 0,
            "section": "Intro",
            "section_number": "None"
            }
        ],
        "Section 2": [
            {
            "text": "Another chunk of text in section 2.",
            "pages": "3-3",
            "chunk_index": 22, # Chunk indices are global and do not get re-started per section
            "section": "Section 2",
            "section_number": "None"
            }
        ]
    }

@pytest.fixture
def mock_grobid_json_flat_content():
    return [
        {
            "text": "The global burden of diarrhoeal disease, the fourth leading cause of under-5 mortality, is likely to increase because of climate change.",
            "pages": "1-1",
            "chunk_index": 0,
            "section": "Intro",
            "section_number": "None"
        },
        {
            "text": "Another chunk of text in section 2.",
            "pages": "3-3",
            "chunk_index": 22, # Chunk indices are global and do not get re-started per section
            "section": "Section 2",
            "section_number": "None"
        }
    ]

@pytest.fixture
def mock_document_pdf(mock_pdf_path):
    with patch('minedd.document.init_marker') as mock_init_marker:
        doc = DocumentPDF(pdf_path=mock_pdf_path, marker_converter=mock_init_marker)
        yield doc

class TestDocumentPDF:
    def test_init(self, mock_document_pdf, mock_pdf_path):
        assert mock_document_pdf.pdf_path == mock_pdf_path
        assert mock_document_pdf.markdown is None
        assert mock_document_pdf.json_content is None
        assert mock_document_pdf.tables == []

    def test_from_json_success(self, mock_json_path, mock_md_content, mock_grobid_json_per_section_content):
        json_content = {
            "original_path": "/fake/path/to/document.pdf",
            "markdown": mock_md_content,
            "text_chunks": {
                "title": "Paper Title",
                "grouped_by_section": True,
                "sections_titles": ["Intro", "Section 2"],
                "sections_content": mock_grobid_json_per_section_content,
                "authors": ["Author1", "Author2"], # Optional field
                "abstract": "Abstract of paper extracted by Grobid", # Optional field
                "publication_date": "3 June 2019", # Optional field
            },
            "tables_as_json": [{"col1": [1], "col2": [2]}]
        }
        with patch('builtins.open', mock_open(read_data=json.dumps(json_content))) as mock_file:
            doc = DocumentPDF.from_json(mock_json_path)
            mock_file.assert_called_once_with(mock_json_path)
            assert doc is not None
            assert doc.pdf_path == json_content["original_path"]
            assert doc.markdown == json_content["markdown"]
            assert doc.json_content == json_content["text_chunks"]
            assert len(doc.tables) == 1
            assert isinstance(doc.tables[0], pd.DataFrame)
            assert doc.tables[0].equals(pd.DataFrame(json_content["tables_as_json"][0]))

    def test_from_json_file_not_found(self, mock_json_path, capsys):
        with patch('builtins.open', side_effect=FileNotFoundError):
            doc = DocumentPDF.from_json(mock_json_path)
            assert doc is None
            captured = capsys.readouterr()
            assert f"File not found: {mock_json_path}" in captured.out

    def test_get_markdown_already_loaded(self, mock_document_pdf):
        mock_document_pdf.markdown = "pre-loaded markdown"
        markdown = mock_document_pdf.get_markdown()
        assert markdown == "pre-loaded markdown"


    def test_get_chunks_chars(self, mock_document_pdf):
        mock_document_pdf.markdown = "This is a test."
        chunks = mock_document_pdf.get_chunks(mode='chars')
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_get_chunks_from_json(self, mock_document_pdf):
        mock_document_pdf.json_content = {
            "title": "Test Title",
            "grouped_by_section": False,
            "chunks": [{"text": "sentence 1", "pages": "1", "chunk_index": 0, "section": "Intro"}]
        }
        with patch('minedd.document.process_from_grobid_chunks') as mock_process:
            mock_process.return_value = ["chunk1"]
            chunks = mock_document_pdf.get_chunks(mode='from_json')
            assert chunks == ["chunk1"]

    def test_to_json(self, mock_document_pdf, mock_json_path):
        mock_document_pdf.markdown = "md"
        mock_document_pdf.json_content = "json"
        mock_document_pdf.tables = [pd.DataFrame({'a': [1]})]
        with patch('builtins.open', mock_open()) as mock_file:
            mock_document_pdf.to_json(mock_json_path)
            mock_file.assert_called_once_with(mock_json_path, 'w', encoding='utf-8')


class TestDocumentMarkdown:
    def test_init_with_content(self, mock_md_content):
        doc = DocumentMarkdown(md_content=mock_md_content)
        assert doc.markdown == mock_md_content

    def test_init_with_path(self, mock_md_content):
        with patch('builtins.open', mock_open(read_data=mock_md_content)) as mock_file:
            doc = DocumentMarkdown(md_path="/fake/path.md")
            mock_file.assert_called_once_with("/fake/path.md", 'r', encoding='utf-8')
            assert doc.markdown == mock_md_content

    def test_init_no_input(self):
        with pytest.raises(ValueError):
            DocumentMarkdown()

    def test_get_title(self, mock_md_content):
        doc = DocumentMarkdown(md_content=mock_md_content)
        assert doc.get_title() == "Title"

    def test_get_markdown(self, mock_md_content):
        doc = DocumentMarkdown(md_content=mock_md_content)
        assert "markdown content" in doc.get_markdown()

    def test_convert_to_chunks(self, mock_md_content):
        doc = DocumentMarkdown(md_content=mock_md_content)
        chunks = doc.convert_to_chunks()
        assert isinstance(chunks, list)
        assert len(chunks) > 0

def test_process_from_grobid_chunks():
    json_content = {
        "title": "Test Title",
        "grouped_by_section": True,
        "sections_content": {
            "Intro": [{"text": "s1", "pages": "1", "chunk_index": 0, "section": "Intro"}]
        }
    }
    docs = process_from_grobid_chunks("file.pdf", json_content, as_langchain_docs=True)
    assert len(docs) == 1
    assert docs[0].metadata['title'] == "Test Title"

def test_get_documents_from_directory():
    with patch('os.listdir', return_value=['doc1.json']):
        with patch('minedd.document.DocumentPDF.from_json') as mock_from_json:
            mock_doc = MagicMock()
            mock_doc.get_chunks.return_value = ["chunk1"]
            mock_from_json.return_value = mock_doc
            
            docs = get_documents_from_directory("/fake/dir")
            assert docs == ["chunk1"]

