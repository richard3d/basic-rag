import aiofiles
from langchain_text_splitters import RecursiveCharacterTextSplitter

async def chunk_text_file(file_path, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    block_size = chunk_size * 2
    buffer = ""

    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                block = await f.read(block_size)
                if not block:
                    break
                buffer += block
                chunks = splitter.split_text(buffer)
                # Yield all but the last chunk — it may grow when the next block arrives
                for chunk in chunks[:-1]:
                    yield chunk
                # Carry the last chunk forward as the start of the next buffer
                buffer = chunks[-1] if chunks else buffer
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    # Flush whatever remains once EOF is reached
    if buffer:
        for chunk in splitter.split_text(buffer):
            yield chunk
