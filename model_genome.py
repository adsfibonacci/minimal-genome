class Genome:
    def __init__(self, n):
        self.n = n
        self.data = [1] * n
        self.valid_bounds = [(0, n)]
        return

    def mask(self, a, b=None):
        a %= self.n
        if b is None:
            if self.data[a] == 0:
                raise ValueError(f"Gene at index {a} is already masked")
            else:
                self.data[a] = 0
                pass
        else:
            b %= self.n
            
            if a <= b:
                indices = range(a, b+1)
            else:
                indices = list(range(a, self.n)) + list(range(0, b+1))
                pass
            for i in indices:
                if self.data[i] == 0:
                    raise ValueError(f"Gene at index {i} is already masked")
                pass
            for i in indices:
                self.data[i] = 0
                pass
            pass
        self._compute_bounds()
        return
    def _compute_bounds(self):
        n = self.n
        bounds = []
        start = None
        if all([ self.data[i] == 0 for i in range(n)]):
            self.valid_bounds = []
            return
        for i in range(n):
            if self.data[i] == 1:
                if start is None:
                    start = i
                    pass
                pass
            else:
                if start is not None:
                    end = i - 1
                    bounds.append((start, end))
                    start = None
                    pass
                pass
            pass
        if start is not None:
            if bounds and bounds[0][0] == 0:
                bounds[0] = (start, bounds[0][1])
                pass
            else:
                end = (start + n - 1) % n
                if start == n-1:
                    bounds.append((start, start))
                elif start == end:
                    bounds.append((start, start))
                    pass
                else:
                    bounds.append((start, end))
                    pass
                pass
            pass
        if not bounds:
            bounds.append((0, n - 1))
            pass
        self.valid_bounds = bounds
        return
    
    def _copy_from_bounds(self, bounds_list):
        """Create a new Genome object where only genes in bounds_list are active."""
        new_genome = Genome(self.n)
        new_genome.data = [0] * self.n  

        for start, end in bounds_list:
            if start <= end:
                for i in range(start, end + 1):
                    new_genome.data[i] = 1
                    pass
            else:
                for i in list(range(start, self.n)) + list(range(0, end + 1)):
                    new_genome.data[i] = 1
                    pass
                pass
            pass
        new_genome._compute_bounds()
        return new_genome

    def remove_valid_subsets(self):
        if all([ self.data[i] == 0 for i in range(self.n)]):
            return []
        all_new_bounds_set = set()  # use a set to avoid duplicates
        
        for idx, (start, end) in enumerate(self.valid_bounds):
            n = self.n
            if start <= end:
                active_indices = list(range(start, end + 1))
            else:
                active_indices = list(range(start, n)) + list(range(0, end + 1))
                pass
            L = len(active_indices)
            for i in range(L):
                for j in range(i, L):
                    remove_indices = active_indices[i:j+1]
                    remaining_indices = [x for x in active_indices if x not in remove_indices]
                    if not remaining_indices:
                        continue 
                    temp_bounds = []
                    seq_start = remaining_indices[0]
                    prev = seq_start
                    for x in remaining_indices[1:] + [None]:  # sentinel
                        if x is None or x != (prev + 1) % n:
                            seq_end = prev
                            temp_bounds.append((seq_start, seq_end))
                            if x is not None:
                                seq_start = x
                                pass
                            pass
                        prev = x
                        pass
                    new_bounds = self.valid_bounds[:idx] + self.valid_bounds[idx+1:] + temp_bounds
                    new_bounds = tuple(sorted(new_bounds, key=lambda x: x[0]))
                    all_new_bounds_set.add(new_bounds)
                    pass
                pass
            pass
        next_genomes = [self._copy_from_bounds(list(b)) for b in all_new_bounds_set]
        return next_genomes
    def __repr__(self):
        return f"Genome({self.data})"

    pass


g = Genome(1000)
g.mask(3,142)
g.mask(288)

tree = g.remove_valid_subsets()
print(len(tree))
# for t in tree:
#     print(t)
