from rag import RAGService


if __name__ == "__main__":
    question = (
        "Tell me about differences in annual leave between ApexTech Solutions, "
        "NovaSoft Systems, and Quantum Edge Pvt Ltd."
    )

    rag_service = RAGService()
    result = rag_service.ask(question)

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result["answer"])

    if result["sources"]:
        print("\nSOURCES")
        for source in result["sources"]:
            print(f"- {source['source']}, page {source['page']}")
