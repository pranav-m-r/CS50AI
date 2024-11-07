import sys

from crossword import *


class CrosswordCreator:

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy() for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont

        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size, self.crossword.height * cell_size),
            "black",
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border, i * cell_size + cell_border),
                    (
                        (j + 1) * cell_size - cell_border,
                        (i + 1) * cell_size - cell_border,
                    ),
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (
                                rect[0][0] + ((interior_size - w) / 2),
                                rect[0][1] + ((interior_size - h) / 2) - 10,
                            ),
                            letters[i][j],
                            fill="black",
                            font=font,
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        domain = self.domains.copy()
        for var in domain:
            for word in domain[var].copy():
                # Remove words not satisfying the unary constraint
                if len(word) != var.length:
                    self.domains[var].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        x_domain = self.domains[x].copy()
        y_domain = self.domains[y].copy()
        intersection = self.crossword.overlaps[x, y]
        # No changes made if there is no overlap
        if intersection is None:
            return False

        revised = False
        # For each value in domain of x check for a possible value of y
        for x_val in x_domain:
            found = False
            for y_val in y_domain:
                # Corresponding value for y found
                if x_val[intersection[0]] == y_val[intersection[1]]:
                    found = True
            if not found:
                # Remove the value if there is no corresponding value for y
                self.domains[x].remove(x_val)
                revised = True

        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # If arcs is None, begin with initial list of all arcs
        if arcs is None:
            arcs = list(self.crossword.overlaps.keys())

        while arcs != []:
            (x, y) = arcs.pop(0)
            if self.revise(x, y):
                # If domain is empty after the revision, it is not consistent
                if len(self.domains[x]) == 0:
                    return False
                # Add arcs dependent on the domain of x for a possible issue
                for z in self.crossword.neighbors(x):
                    if z != y:
                        arcs.append((z, x))

        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for var in self.crossword.variables:
            # Check if any variable is still unassigned
            if var not in assignment:
                return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        words = list(assignment.values())
        # Check for duplicate assignments
        if len(words) != len(set(words)):
            return False
        # Check for violation of unary constraints
        for var in assignment:
            if len(assignment[var]) != var.length:
                return False
        # Check for violation of binary constraints
        intersections = self.crossword.overlaps
        for x, y in intersections.keys():
            if x in assignment and y in assignment:
                if intersections[x, y] is None:
                    continue
                x_coord, y_coord = intersections[x, y]
                if assignment[x][x_coord] != assignment[y][y_coord]:
                    return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        ordered_domain = []
        # Get all unassigned neighbors of the variable
        unassigned_neighbors = [
            item for item in self.crossword.neighbors(var) if item not in assignment
        ]

        for value in self.domains[var]:
            ruled_out = 0
            for neighbor in unassigned_neighbors:
                overlap = self.crossword.overlaps[var, neighbor]
                for other_value in self.domains[neighbor]:
                    # Rule out values in the neighbor's domain
                    if value[overlap[0]] != other_value[overlap[1]]:
                        ruled_out += 1
            ordered_domain.append((value, ruled_out))
        # Sort domain by fewest number of values ruled out
        ordered_domain = sorted(ordered_domain, key=lambda value: value[1])

        return [ordered_domain[i][0] for i in range(len(ordered_domain))]

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        eligible_vars = []
        # Get all the unassigned variables
        for var in self.crossword.variables:
            if var not in assignment:
                eligible_vars.append(
                    (var, len(self.domains[var]), -len(self.crossword.neighbors(var)))
                )
        # Sort the variables by the fewest number of remaining values and then by highest degree
        eligible_vars = sorted(eligible_vars, key=lambda value: (value[1], value[2]))

        return eligible_vars[0][0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        # Return the assignment if it is complete
        if self.assignment_complete(assignment):
            return assignment
        # Select an unassigned variable
        var = self.select_unassigned_variable(assignment)
        # Try valuues in the domain of the variable
        for value in self.domains[var]:
            new_assignment = assignment.copy()
            new_assignment[var] = value
            if self.consistent(new_assignment):
                assignment[var] = value
                # Recursively assign other variables with a consistent assignment for current variable
                result = self.backtrack(assignment)
                if result != None:
                    return result

        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
