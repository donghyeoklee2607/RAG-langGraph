def format_docs(docs):
    return "\n".join(
        [
            f"<document><content>{doc.page_content}</content><source>{doc.metadata['source']}</source><page>{int(doc.metadata['page'])+1}</page></document>"
            for doc in docs
        ]
    )


def format_searched_docs(docs):
    return "\n".join(
        [
            f"<document><content>{doc['content']}</content><source>{doc['url']}</source></document>"
            for doc in docs
        ]
    )


def format_task(tasks):
    task_time_pairs = []

    for item in tasks:
        task, time_str = item.rsplit(":", 1)
        time = int(time_str.replace("Time", "").strip())
        task_time_pairs.append((task, time))

    return task_time_pairs
