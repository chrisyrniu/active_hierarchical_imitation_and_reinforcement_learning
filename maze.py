# Modified from https://github.com/boppreh/maze
import random
N, S, W, E = ('n', 's', 'w', 'e')
class Cell(object):
    def __init__(self, x, y, walls):
        self.x = x
        self.y = y
        self.walls = set(walls)

    def __repr__(self):
        # <15, 25 (es  )>
        return '<{}, {} ({:4})>'.format(self.x, self.y, ''.join(sorted(self.walls)))

    def __contains__(self, item):
        # N in cell
        return item in self.walls

    def is_full(self):
        return len(self.walls) == 4

    def _wall_to(self, other):
        assert abs(self.x - other.x) + abs(self.y - other.y) == 1, '{}, {}'.format(self, other)
        if other.y < self.y:
            return N
        elif other.y > self.y:
            return S
        elif other.x < self.x:
            return W
        elif other.x > self.x:
            return E
        else:
            assert False

    def connect(self, other):
        other.walls.remove(other._wall_to(self))
        self.walls.remove(self._wall_to(other))

class Maze:
    def __init__(self, width=4, height=4):
        self.width = width
        self.height = height
        self.win_w = 17
        self.win_h = 17
        self.cells = []
        for y in range(self.height):
            for x in range(self.width):
                self.cells.append(Cell(x, y, [N, S, E, W]))

    def __getitem__(self, index):
        x, y = index
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.cells[x + y * self.width]
        else:
            return None

    def neighbors(self, cell):
        x = cell.x
        y = cell.y
        for new_x, new_y in [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]:
            neighbor = self[new_x, new_y]
            if neighbor is not None:
                yield neighbor

    def randomize(self):
        """
        See http://mazeworks.com/mazegen/mazetut/index.htm
        """
        # Depth first search algorithm
        # TODO change to Prim or Kruscal
        cell_stack = []
        cell = random.choice(self.cells)
        n_visited_cells = 1

        while n_visited_cells < len(self.cells):
            neighbors = [c for c in self.neighbors(cell) if c.is_full()]
            if len(neighbors):
                neighbor = random.choice(neighbors)
                cell.connect(neighbor)
                cell_stack.append(cell)
                cell = neighbor
                n_visited_cells += 1
            else:
                cell = cell_stack.pop()

    def to_walls(self):
        # Only build inner walls
        hww = 0.5 # Half wall width
        cw = self.win_w / self.width
        ch = self.win_h / self.height
        hcw = self.win_w / self.width / 2 # Half cell width
        hch = self.win_h / self.height / 2 # Half cell height
        hw = self.win_w / 2
        hh = self.win_h / 2
        vpos = []
        vsize = []
        # Columns
        for i in range(1,self.width):
            for j in range(self.height):
                cell = self[i,j]
                if W in cell.walls:
                    pos = [i*cw-hw, j*ch+hch-hh]
                    size = [hww, hch+hww]
                    vpos.append(pos)
                    vsize.append(size)
                    assert(E in self[i-1,j].walls)
        # Rows
        for j in range(1,self.height):
            for i in range(self.width):
                cell = self[i,j]
                if N in cell.walls:
                    pos = [i*cw+hcw-hw, j*ch-hh]
                    size = [hcw+hww, hww]
                    vpos.append(pos)
                    vsize.append(size)
                    assert(S in self[i,j-1].walls)
        return vpos, vsize


if __name__=="__main__":
    maze = Maze(4,4)
    maze.randomize()
    vpos,vsize = maze.to_walls()
    print(len(vpos))
    print(len(vsize))
