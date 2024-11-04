import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = {}
    linked_pages = corpus[page]
    num_links = len(linked_pages)
    num_pages = len(corpus)

    if num_links == 0:
        # If no links, distribute probability uniformly across all pages
        for linked_page in corpus:
            distribution[linked_page] = 1 / num_pages
    else:
        # Distribute probability uniformly across all pages using damping factor
        for linked_page in corpus:
            distribution[linked_page] = (1 - damping_factor) / num_pages
        # Distribute probability across linked pages using damping factor
        for linked_page in linked_pages:
            distribution[linked_page] += damping_factor / num_links
    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = list(corpus.keys())
    page = pages[random.randint(0, len(pages) - 1)]
    pagerank = {page: 0 for page in pages}

    # Sample n pages
    for _ in range(n):
        transition_prob = transition_model(corpus, page, damping_factor)
        random_value = random.random()
        cumulative_probability = 0
        # Pick the sample using a random value
        for current_page, probability in transition_prob.items():
            cumulative_probability += probability
            if cumulative_probability > random_value:
                page = current_page
                break
        pagerank[page] += 1

    # Normalize the pagerank values
    pagerank = {page: rank / n for page, rank in pagerank.items()}

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    pagerank = {page: 1 / N for page in corpus}
    new_pagerank = {page: 0 for page in corpus}

    while True:
        # Calculate new pagerank values
        for page in new_pagerank:
            sum_pr = sum(
                pagerank[i] / len(corpus[i]) for i in corpus if page in corpus[i]
            )
            sum_pr += sum(pagerank[i] / N for i in corpus if len(corpus[i]) == 0)
            new_pagerank[page] = (1 - damping_factor) / N + damping_factor * sum_pr

        # Normalize the pagerank values
        n = sum(new_pagerank.values())
        new_pagerank = {page: rank / n for page, rank in new_pagerank.items()}

        # Check for convergence
        if all(
            abs(new_pagerank[page] - pagerank[page]) < 0.001 for page in new_pagerank
        ):
            break

        pagerank = new_pagerank.copy()

    return new_pagerank


if __name__ == "__main__":
    main()
