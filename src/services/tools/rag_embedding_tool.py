import os

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from src.services.langgraph_handler import LangGraphHandler


class CreateRAGData:
    def __init__(self):
        self.to_process_path = Path() / "to_process"
        self.processed_path = Path() / "processed"
        self.files = self.list_pdf_files_in_dir()

    def list_pdf_files_in_dir(self):
        """
        List all files in the DIR_TO_PROCESS directory.
        """
        print(f"Listing files in {self.to_process_path}")
        files = os.listdir(self.to_process_path)
        files_pdf = [file for file in files if file.endswith("pdf")]
        print(f"Found {len(files_pdf)} PDF files in {self.to_process_path}")
        return files_pdf

    def extract_text_from_pdf(self, file_name: str) -> list[Document]:
        """
        Extract text from a PDF file.
        """
        print(f"Extracting text from {file_name}")
        loader = PyMuPDFLoader(os.path.join(self.to_process_path, file_name))
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = loader.load()
        return splitter.split_documents(docs)

    def move_file_to_processed(self, file_name: str) -> None:
        """
        Move the processed file to the processed directory.
        """
        os.rename(
            os.path.join(self.to_process_path, file_name),
            os.path.join(self.processed_path, file_name),
        )
        print(f"Moved {file_name} from {self.to_process_path} to {self.processed_path}")

    def main(self) -> None:
        """
        Main function to process the PDF files.
        """
        for file in self.files:
            print("-" * 20)
            print(f"Processing {file}")
            print("-" * 20)
            split_docs = self.extract_text_from_pdf(file)
            lg_handler = LangGraphHandler(9999999999)
            lg_handler.add_to_vector_store_support_docs(split_docs)
            self.move_file_to_processed(file)
            print(f"Processed {file}")
            print("-" * 20)
            print("\n\n")


if __name__ == "__main__":
    rag_data = CreateRAGData()
    rag_data.main()
