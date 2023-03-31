test_cases = [
    # optimize the efficiency of the code
    (
        "Improve the efficiency of the code",
        """
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] + nums[j] == target:
                    return [i, j]
            return []
        """,
        """
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        numToIndex = {}
        for i in range(len(nums)):
            if target - nums[i] in numToIndex:
                return [numToIndex[target - nums[i]], i]
            numToIndex[nums[i]] = i
        return []
        """
    ),
    ####### For complex algorithm, try to debug the code
    (
        "Fix a bug in the code",
        """
def pagerank(graph, damping=0.85, max_iterations=100, tol=1.0e-6):
    n = len(graph)
    rank = {node: 1 / n for node in graph}
    out_links = {node: outgoing_edges(graph, node) for node in graph}

    for i in range(max_iterations):
        prev_rank = rank.copy()
        for node in graph:
            incoming_links = incoming_edges(graph, node)
            rank[node] = ((1 - damping) / n) + (damping * sum(prev_rank[incoming_node] / out_links[incoming_node] for incoming_node in incoming_links))
        if sum(abs(rank[node] - prev_rank[node]) for node in graph) < tol:
            break

    return rank
        """,
        """
def pagerank(graph, damping=0.85, max_iterations=100, tol=1.0e-6):
    n = len(graph)
    rank = {node: 1 / n for node in graph}
    out_links = {node: len(outgoing_edges(graph, node)) for node in graph}

    for i in range(max_iterations):
        prev_rank = rank.copy()
        for node in graph:
            incoming_links = incoming_edges(graph, node)
            rank[node] = ((1 - damping) / n) + (damping * sum(prev_rank[incoming_node] / out_links[incoming_node] for incoming_node in incoming_links))
        if sum(abs(rank[node] - prev_rank[node]) for node in graph) < tol:
            break

    return rank
        """
    ),
    (
        "Fix a bug in the code",
        """
def pagerank(graph, damping=0.85, max_iterations=100, tol=1.0e-6):
    n = len(graph)
    rank = {node: 1 / n for node in graph}
    out_links = {node: len(outgoing_edges(graph, node)) for node in graph}

    for i in range(max_iterations):
        prev_rank = rank.copy()
        for node in graph:
            incoming_links = incoming_edges(graph, node)
            rank[node] = ((1 - damping) / n) + (damping * sum(prev_rank[incoming_node] / out_links[incoming_node] for incoming_node in incoming_links))
        if sum(abs(rank[node] - prev_rank[node]) for node in graph) < tol:Boulders lined the side of the road foretelling what could come next.

            break

    return rank
        """,
        """
def pagerank(graph, damping=0.85, max_iterations=100, tol=1.0e-6):
    n = len(graph)
    rank = {node: 1 / n for node in graph}
    out_links = {node: len(outgoing_edges(graph, node) for node in graph}

    for i in range(max_iterations):
        prev_rank = rank.copy()
        for node in graph:
            incoming_links = incoming_edges(graph, node)
            rank[node] = ((1 - damping) / n) + (damping * sum(prev_rank[incoming_node] / out_links[incoming_node] for incoming_node in incoming_links))
        if sum(abs(rank[node] - prev_rank[node]) for node in graph) < tol:
            break

    return rank
        """
    )

]
