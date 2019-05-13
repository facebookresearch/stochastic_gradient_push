# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Graph Manager Class

:description: Class provides an API for loading different peer-to-peer
    communication topologies, and cycling through peers.
"""

from math import log as mlog


class GraphManager(object):

    def __init__(self, rank, world_size, peers_per_itr=1):
        assert int(peers_per_itr) >= 1
        self.rank = rank
        self.world_size = world_size
        self.phone_book = self._make_graph()
        self._peers_per_itr = peers_per_itr
        self._group_indices = [i for i in range(peers_per_itr)]

    @property
    def peers_per_itr(self):
        return self._peers_per_itr

    @peers_per_itr.setter
    def peers_per_itr(self, v):
        self._peers_per_itr = v
        # set group-indices attr. --- point to out-peers in phone-book
        self._group_indices = [i for i in range(v)]

    def _make_graph(self):
        """
        Returns a nested list of peers; the outer-list is indexed by rank,
        the inner list denotes the set of peers that 'rank' can send
        messages to at any point in time
        """
        raise NotImplementedError

    def is_regular_graph(self):
        """ Whether each node has the same number of in-peers as out-peers """
        raise NotImplementedError

    def is_bipartite_graph(self):
        """ Whether graph is bipartite or not """
        raise NotImplementedError

    def is_passive(self, rank=None):
        """ Whether 'rank' is a passive node or not """
        raise NotImplementedError

    def is_dynamic_graph(self, graph_type=None):
        """ Whether the graph-type is dynamic (as opposed to static) """
        raise NotImplementedError

    def get_peers(self, rotate=False):
        """ Returns the out and in-peers corresponding to 'self.rank' """
        # cycle through in- and out-peers by updating group-index
        if rotate:
            self._rotate_group_indices()

        # get out- and in-peers using new group-indices
        out_peers, in_peers = [], []
        for group_index in self._group_indices:
            out_peers.append(self.phone_book[self.rank][group_index])
            for rank, peers in enumerate(self.phone_book):
                if rank == self.rank:
                    continue
                if self.rank == peers[group_index]:
                    in_peers.append(rank)
        return out_peers, in_peers

    def _rotate_group_indices(self):
        """ Incerement group indices to point to the next out-peer """
        increment = self.peers_per_itr
        for i, group_index in enumerate(self._group_indices):
            self._group_indices[i] = int((group_index + increment)
                                         % len(self.phone_book[self.rank]))

    def _rotate_forward(self, r, p):
        """ Helper function returns peer that is p hops ahead of r """
        return (r + p) % self.world_size

    def _rotate_backward(self, r, p):
        """ Helper function returns peer that is p hops behind r """
        temp = r
        for _ in range(p):
            temp -= 1
            if temp < 0:
                temp = self.world_size - 1
        return temp


class DynamicDirectedExponentialGraph(GraphManager):

    def _make_graph(self):
        phone_book = [[] for _ in range(self.world_size)]
        for rank in range(self.world_size):
            group = phone_book[rank]
            for i in range(0, int(mlog(self.world_size - 1, 2)) + 1):
                f_peer = self._rotate_forward(rank, 2 ** i)
                if f_peer not in group:
                    group.append(f_peer)
                b_peer = self._rotate_backward(rank, 2 ** i)
                if b_peer not in group:
                    group.append(b_peer)
        return phone_book

    def is_regular_graph(self): return True

    def is_bipartite_graph(self): return False

    def is_passive(self, rank=None): return False

    def is_dynamic_graph(self, graph_type=None): return True


class DynamicBipartiteExponentialGraph(GraphManager):

    def _make_graph(self):
        phone_book = [[] for _ in range(self.world_size)]
        for rank in range(self.world_size):
            group = phone_book[rank]
            for i in range(0, int(mlog(self.world_size - 1, 2)) + 1):
                if i == 0:
                    f_peer = self._rotate_forward(rank, 1)
                    b_peer = self._rotate_backward(rank, 1)
                else:
                    f_peer = self._rotate_forward(rank, 1 + 2 ** i)
                    b_peer = self._rotate_backward(rank, 1 + 2 ** i)
                # create directory for non-passive peers
                if not self.is_passive(rank) and (
                   self.is_passive(f_peer) and self.is_passive(b_peer)):
                    if f_peer not in group:
                        group.append(f_peer)  # forward peer...
                    if b_peer not in group:
                        group.append(b_peer)  # then backward peer
                # create directory for passive peers
                elif self.is_passive(rank) and (
                   not (self.is_passive(f_peer) or self.is_passive(b_peer))):
                    if b_peer not in group:
                        group.append(b_peer)  # backward peer...
                    if f_peer not in group:
                        group.append(f_peer)  # then forward peer
        return phone_book

    def is_regular_graph(self): return True

    def is_bipartite_graph(self): return True

    def is_passive(self, rank=None):
        rank = self.rank if rank is None else rank
        return (rank % 2) == 0

    def is_dynamic_graph(self, graph_type=None): return True


class DynamicDirectedLinearGraph(GraphManager):

    def _make_graph(self):
        phone_book = [[] for _ in range(self.world_size)]
        for rank in range(self.world_size):
            group = phone_book[rank]
            for i in range(1, self.world_size):
                if i % 2 == 0:
                    continue
                f_peer = self._rotate_forward(rank, i)
                if f_peer not in group:
                    group.append(f_peer)
                b_peer = self._rotate_backward(rank, i)
                if b_peer not in group:
                    group.append(b_peer)
        return phone_book

    def is_regular_graph(self): return True

    def is_bipartite_graph(self): return False

    def is_passive(self, rank=None): return False

    def is_dynamic_graph(self, graph_type=None): return True


class DynamicBipartiteLinearGraph(GraphManager):

    def _make_graph(self):
        phone_book = [[] for _ in range(self.world_size)]
        for rank in range(self.world_size):
            group = phone_book[rank]
            for i in range(1, self.world_size):
                f_peer = self._rotate_forward(rank, i)
                b_peer = self._rotate_backward(rank, i)
                # create directory for non-passive peers
                if not self.is_passive(rank) and (
                   self.is_passive(f_peer) and self.is_passive(b_peer)):
                    if f_peer not in group:
                        group.append(f_peer)  # forward peer...
                    if b_peer not in group:
                        group.append(b_peer)  # then backward peer
                # create directory for passive peers
                elif self.is_passive(rank) and (
                   not (self.is_passive(f_peer) or self.is_passive(b_peer))):
                    if b_peer not in group:
                        group.append(b_peer)  # backward peer...
                    if f_peer not in group:
                        group.append(f_peer)  # then forward peer
        return phone_book

    def is_regular_graph(self): return True

    def is_bipartite_graph(self): return True

    def is_passive(self, rank=None):
        rank = self.rank if rank is None else rank
        return (rank % 2) == 0

    def is_dynamic_graph(self, graph_type=None): return True
