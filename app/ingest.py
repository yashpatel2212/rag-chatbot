import fitz

def extract_pages(pdf_path: str) -> list[dict[str,int| str]]:

    pages = []
    total_characters = 0

    with fitz.open(pdf_path) as document:
        print(f"\nExtracting pages from: {pdf_path.name}")
        print(f"Total Pages: {len(document)}")

        for page_number, page in enumerate(document):

            text = page.get_text().strip()

            if not text:
                continue

            total_characters += len(text)

            pages.append(
                {
                    "page_number": page_number + 1,
                    "text": text
                }
            )

    

    print(f"\nTotal Characters: {total_characters}")

    return pages
