from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE , CHUNK_OVERLAP

def chunk_text(text : str) -> list[str]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_text(text)

    return chunks

# if __name__ == "__main__":
#     sample_text = "This is a sample text that will be chunked into smaller pieces. " * 20
#     chunks = chunk_text(sample_text)
#     for i, chunk in enumerate(chunks):
#         print(f"Chunk {i+1}: {chunk}\n")