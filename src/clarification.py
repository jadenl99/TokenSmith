import textwrap
from typing import List, Optional, Tuple

from src.generator import run_llama_cpp, text_cleaning


CLEAR_TAG = "CLEAR"
AMBIGUOUS_TAG = "AMBIGUOUS"


def _format_chunks_for_prompt(
    chunks: List[str],
    max_chunks: int = 5,
    max_chars_per_chunk: int = 500,
) -> str:
    """
    Render a short, numbered preview of the retrieved chunks for the prompt.
    Chunks are truncated so the classifier stays cheap and focused on topics
    rather than details.
    """
    rendered: List[str] = []
    for i, chunk in enumerate(chunks):
        if i >= max_chunks:
            break
        if chunk is None:
            continue
        text = str(chunk).strip()
        if not text:
            continue
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rstrip() + "..."
        rendered.append(f"[Chunk {i + 1}]\n{text}")
    if not rendered:
        return "(no chunks retrieved)"
    return "\n\n".join(rendered)


def classify_and_clarify(
    query: str,
    chunks: List[str],
    model_path: str,
    max_tokens: int = 96,
    **llm_kwargs,
) -> Tuple[bool, Optional[str]]:
    """
    Classify a query as ambiguous or clear, conditioned on the chunks that
    the retriever returned for it. If ambiguous, also return a single
    follow-up question the user can answer to disambiguate.
    """
    chunks_block = _format_chunks_for_prompt(chunks)
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a strict query triage assistant for a database systems textbook Q&A tool.
        You are given the user's question and the top chunks a retriever pulled
        from the textbook for that question. Decide whether the question is
        specific enough to be answered from those chunks, or whether it is
        ambiguous and needs a clarifying follow-up.

        Mark the question AMBIGUOUS if ANY of the following hold in light of
        the retrieved chunks:
        - The chunks span multiple distinct subtopics and it is not clear
          which one the user means (e.g. chunks about B+ trees, hash indexes,
          and index selection for a question that just says "indexing").
        - None of the chunks actually address the question — the retriever
          grabbed loosely related passages because the query was too vague or
          under-scoped.
        - The question contains an unresolved pronoun or demonstrative
          ("it", "this", "the algorithm") and the chunks do not pin down a
          single referent.
        - The question is a bare noun phrase or single keyword, or asks for
          an example/comparison without saying of what.
        - Multiple chunks could each plausibly be "the" answer but they
          describe incompatible concepts.

        Mark the question CLEAR only if at least one retrieved chunk clearly
        addresses the specific thing being asked, AND the chunks agree on
        what topic is in scope.

        When in doubt, prefer AMBIGUOUS.

        Respond in EXACTLY one of these two formats, with no extra text:
        CLEAR
        AMBIGUOUS: <one short follow-up question that names 2-3 distinct candidate topics drawn from the retrieved chunks and leaves the door open for "any other topic" so the user is never forced to pick from the suggestions; vary the phrasing naturally across cases rather than always using the same dash-separated list>
        <|im_end|>
        <|im_start|>user
        Question: How does strict two-phase locking ensure conflict serializability?

        Retrieved chunks:
        [Chunk 1]
        Strict two-phase locking (Strict 2PL) requires that a transaction hold all
        of its exclusive locks until it commits or aborts. Because no other
        transaction can read or write a value written by T until T finishes, the
        resulting schedule is conflict serializable and also recoverable.

        [Chunk 2]
        Under 2PL, every transaction has a growing phase (acquiring locks) and a
        shrinking phase (releasing locks). Once a transaction has released any
        lock, it may not acquire another. This rule is sufficient to guarantee
        conflict serializability.

        [Chunk 3]
        Lock compatibility: shared locks are compatible with other shared locks
        but not with exclusive locks; exclusive locks are incompatible with all
        other lock modes.
        <|im_end|>
        <|im_start|>assistant
        CLEAR<|im_end|>
        <|im_start|>user
        Question: Tell me about indexing.

        Retrieved chunks:
        [Chunk 1]
        A B+ tree is a balanced search tree in which every leaf is at the same
        depth. Internal nodes store separator keys while leaves store the
        indexed entries, linked together to support range scans.

        [Chunk 2]
        Static hashing maps a search key to a bucket via a hash function. It
        works well for equality lookups but supports range queries poorly and
        suffers from overflow chains as the file grows.

        [Chunk 3]
        A clustered index determines the physical order of rows in the file, so
        a table can have at most one clustered index. Unclustered (secondary)
        indexes instead store pointers to the underlying rows.

        [Chunk 4]
        Index selection is a design-time decision: the DBA picks indexes based
        on the expected workload, balancing lookup speed against the cost of
        maintaining the index on inserts and updates.
        <|im_end|>
        <|im_start|>assistant
        AMBIGUOUS: "Indexing" covers a lot of ground in this chapter — are you interested in how B+ trees are structured, how hash indexes handle lookups, or the tradeoff between clustered and unclustered indexes? Happy to go somewhere else if none of these fit.<|im_end|>
        <|im_start|>user
        Question: Can you explain this concept?

        Retrieved chunks:
        [Chunk 1]
        The relational model organizes data into relations (tables) with a fixed
        schema of typed attributes.

        [Chunk 2]
        Write-ahead logging requires that log records describing a change reach
        stable storage before the corresponding data page is flushed.

        [Chunk 3]
        A foreign key constraint ensures that a value in one relation refers to
        an existing tuple in another relation.
        <|im_end|>
        <|im_start|>assistant
        AMBIGUOUS: It's not clear what "this concept" refers to — the retrieved passages touch on three unrelated ideas. Did you mean (1) the relational model, (2) write-ahead logging, or (3) foreign key constraints? If it's something else entirely, tell me which topic and I'll focus there.<|im_end|>
        <|im_start|>user
        Question: What is the difference between 3NF and BCNF?

        Retrieved chunks:
        [Chunk 1]
        A relation is in third normal form (3NF) if, for every non-trivial
        functional dependency X -> A, either X is a superkey or A is part of
        some candidate key. 3NF tolerates certain dependencies that BCNF does
        not, in exchange for always being achievable via a dependency-preserving
        decomposition.

        [Chunk 2]
        A relation is in Boyce-Codd normal form (BCNF) if, for every non-trivial
        functional dependency X -> A, X is a superkey. BCNF is strictly stronger
        than 3NF but a BCNF decomposition may lose some functional dependencies.
        <|im_end|>
        <|im_start|>assistant
        CLEAR<|im_end|>
        <|im_start|>user
        Question: How does locking work?

        Retrieved chunks:
        [Chunk 1]
        Two-phase locking (2PL) has a growing and shrinking phase; once a
        transaction releases a lock it cannot acquire another.

        [Chunk 2]
        Hierarchical locking introduces intention locks (IS, IX, SIX) so that a
        transaction can lock at a coarse granularity while still allowing
        fine-grained locks on descendants.

        [Chunk 3]
        Deadlock detection builds a waits-for graph over transactions and aborts
        a victim whenever a cycle appears.

        [Chunk 4]
        Optimistic concurrency control avoids locking on reads entirely and
        instead validates transactions at commit time.
        <|im_end|>
        <|im_start|>assistant
        AMBIGUOUS: Locking is a big area; could you narrow it down? I can walk through the two-phase locking protocol, explain how hierarchical/intention locks let you mix coarse and fine granularity, or focus on how deadlocks are detected and resolved — or point me at a different angle if none of those are what you had in mind.<|im_end|>
        <|im_start|>user
        Question: {query}

        Retrieved chunks:
        {chunks_block}
        <|im_end|>
        <|im_start|>assistant
        """)

    prompt = text_cleaning(prompt)
    output = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=max_tokens,
        temperature=0.0,
        **llm_kwargs,
    )

    try:
        text = output["choices"][0]["text"].strip()
    except Exception:
        return False, None

    # Take only the first non-empty line to guard against rambling outputs
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if not first_line:
        return False, None

    upper = first_line.upper()
    if upper.startswith(AMBIGUOUS_TAG):
        remainder = first_line[len(AMBIGUOUS_TAG):].lstrip(" :-").strip()
        if not remainder:
            return False, None
        return True, remainder

    return False, None


def merge_clarification(
        original_query: str,
        clarification: str,
        model_path: str,
        max_tokens: int = 500,
        **llm_kwargs,
    ) -> str:
    """
    Combine the original query with the user's clarification into a single
    standalone query suitable for retrieval. Uses an LLM to produce a clean
    rephrasing instead of a raw concatenation, falling back to a simple join
    on any failure.
    """
    original_query = original_query.strip()
    clarification = clarification.strip()
    if not clarification:
        return original_query
    fallback = f"{original_query} (clarification: {clarification})"

    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a query rewriting assistant for a database systems textbook Q&A tool.
        You are given the user's original question and their short clarification reply.
        Rewrite them into ONE standalone question that:
        - Incorporates the specific topic or scope from the clarification.
        - Reads naturally, as if the user had asked it that way originally.
        - Preserves the intent and question type of the original.
        - Contains no meta phrasing like "clarification:", "user said", or quotes.


        Output ONLY the rewritten question on a single line, with no preamble.


        Examples:
        Original: Tell me about indexing.
        Clarification: B+ trees
        Output: Tell me about B+ tree indexing.


        Original: How does locking work?
        Clarification: specifically deadlock detection
        Output: How does deadlock detection work in database locking?


        Original: Can you explain this concept?
        Clarification: write-ahead logging
        Output: Can you explain write-ahead logging?
        <|im_end|>
        <|im_start|>user
        Original: {original_query}
        Clarification: {clarification}


        Output:
        <|im_end|>
        <|im_start|>assistant
        """)

    prompt = text_cleaning(prompt)
    try:
        output = run_llama_cpp(
            prompt,
            model_path,
            max_tokens=max_tokens,
            temperature=0.1,
            **llm_kwargs,
        )
        rewritten = output["choices"][0]["text"].strip()
    except Exception:
        return fallback

    # Take the first non-empty line to guard against rambling outputs
    first_line = next((ln.strip() for ln in rewritten.splitlines() if ln.strip()), "")
    if not first_line:
        return fallback

    # Guard against hallucinated long outputs
    max_len = max(len(original_query) + len(clarification) + 40, 200)
    if len(first_line) > max_len * 2:
        return fallback

    return first_line