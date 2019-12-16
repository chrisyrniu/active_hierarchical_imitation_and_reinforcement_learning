import sys
import xml.etree.ElementTree as ET

class XMLWrapper:
    def __init__(self, filename):
        self.filename = filename
        self.tree = ET.parse(filename)
        self.root = self.tree.getroot()
        self.worldbody = next(self.root.iter('worldbody'))

    def set_wall(self, name, pos, size):
        found = False
        if len(pos) == 2:
            pos.append(1.0)
            size.append(1.0)
        for node in self.worldbody:
            if node.tag == 'body' and node.get('name') == name:
                child = node[0]
                print(pos)
                print(size)
                node.set('pos', ' '.join(map(str, pos)))
                child.set('size', ' '.join(map(str, size)))
                found = True
        if not found:
            new_body = ET.SubElement(self.worldbody, 'body')
            new_body.set('name', name)
            new_body.set('pos', ' '.join(map(str, pos)))
            str_size = ' '.join(map(str, size))
            child = ET.SubElement(new_body, 'geom')
            child_attr = {'type':'box','size':str_size,'contype':'1','conaffinity':'1', 'rgba':'0.4 0.4 0.4 1'}
            for key,value in child_attr.items():
                child.set(key,value)

    def get_wall(self, name):
        pos = None
        size = None
        for node in self.worldbody:
            if node.tag == 'body' and node.get('name') == name:
                child = node[0]
                pos = list(map(float, node.get('pos').strip().split()))
                size = list(map(float, child.get('size').strip().split()))
                assert(len(pos) == 3)
                pos = pos[:2]
                size = size[:2]
        return pos, size

    def get_walls(self):
        vpos = []
        vsize = []
        for node in self.worldbody:
            if node.tag == 'body' and 'wall' in node.get('name'):
                _name = node.get('name')
                child = node[0]
                if node.get('pos') is None:
                    print(_name)
                pos = list(map(float, node.get('pos').strip().split()))
                size = list(map(float, child.get('size').strip().split()))
                assert(len(pos) == 3)
                pos = pos[:2]
                size = size[:2]
                vpos.append(pos)
                vsize.append(size)
        return vpos, vsize

    def get_inner_walls(self):
        vpos = []
        vsize = []
        for node in self.worldbody:
            if node.tag == 'body' and 'wall' in node.get('name'):
                _name = node.get('name')
                if _name.split('_')[0] in ['east','west','south','north']:
                    continue
                child = node[0]
                pos = list(map(float, node.get('pos').strip().split()))
                size = list(map(float, child.get('size').strip().split()))
                assert(len(pos) == 3)
                pos = pos[:2]
                size = size[:2]
                vpos.append(pos)
                vsize.append(size)
        return vpos, vsize

    def del_wall(self, name):
        for node in self.worldbody:
            if node.tag == 'body' and node.get('name') == name:
                self.worldbody.remove(node)

    def reload(self, filename):
        self.filename = filename
        self.tree = ET.parse(filename)
        self.root = self.tree.getroot()
        self.worldbody = next(self.root.iter('worldbody'))

    def write(self, filename=None):
        if filename is None:
            filename = self.filename
        self.tree.write(filename)
        self.reload(filename)

