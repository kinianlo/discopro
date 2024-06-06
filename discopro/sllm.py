"""
Implementation of the categorical semantics of the SLLM type-logical grammar.
"""
from discopy.utils import factory_name, from_tree
from discopy import biclosed, rigid, messages, monoidal


@monoidal.Diagram.subclass
class Diagram(biclosed.Diagram):
    @staticmethod
    def project(n: int, ty: 'Bang') -> 'Project':
        return Project(n, ty)

    @staticmethod
    def remove_nabla(ty: 'Nabla') -> 'RemoveNabla':
        return RemoveNabla(ty)

    @staticmethod
    def perm_right(nabla: 'Nabla', right: 'Ty') -> 'PermRight':
        return PermRight(nabla, right)

    @staticmethod
    def perm_left(left: 'Ty', nabla: 'Nabla') -> 'PermLeft':
        return PermLeft(left, nabla)

    @staticmethod
    def fa(left: 'Ty', right: 'Ty') -> 'FA':
        """Forward application."""
        return FA(left)

    @staticmethod
    def ba(left: 'Ty', right: 'Ty') -> 'BA':
        """Backward application."""
        return BA(right)


FA = biclosed.FA
BA = biclosed.BA
Box = biclosed.Box


class Ty(biclosed.Ty):
    """
    Objects in a free biclosed monoidal category + SLLM.
    Generated by the following grammar:

        ty ::= Ty(name) | ty @ ty | ty >> ty | ty << ty | Bang(ty) | Nabla(ty)

    Examples
    --------
    >>> x, y = Ty('x'), Ty('y')
    >>> print(Bang(x @ y))
    !(x @ y)
    >>> print(Nabla(x @ y))
    ∇(x @ y)
    >>> print(Bang(Nabla(x)))
    !(∇(x))
    """

    @staticmethod
    def upgrade(old):
        if len(old) == 1 and isinstance(old[0], (Nabla, Bang, Over, Under)):
            return old[0]
        return Ty(*old.objects)

    def __init__(self, *objects, left=None, right=None):
        self.left, self.right = left, right
        super().__init__(*objects)

    def bang(self):
        return Bang(self)

    def nabla(self):
        return Nabla(self)

    def __lshift__(self, other):
        return Over(self, other)

    def __rshift__(self, other):
        return Under(self, other)


class Id(biclosed.Id, Diagram):
    """Implements the identity diagram of a given type.
    >>> s, t = Ty('x', 'y'), Ty('z', 'w')
    >>> f = Box('f', s, t)
    >>> assert f >> Id(t) == f == Id(s) >> f
    """

    def __init__(self, dom=Ty()):
        biclosed.Id.__init__(self, dom)


Diagram.id = Id


class Over(biclosed.Over, Ty):
    def __init__(self, left, right):
        Ty.__init__(self, self)
        biclosed.Over.__init__(self, left, right)


class Under(biclosed.Under, Ty):
    def __init__(self, left, right):
        Ty.__init__(self, self)
        biclosed.Under.__init__(self, left, right)


class UnaryTyConstructor:
    """Ty constructor with one input."""

    def __init__(self, ty: Ty):
        self.ty = ty

    def to_tree(self) -> dict:
        return {"factory": factory_name(self), "ty": self.ty.to_tree()}

    @classmethod
    def from_tree(cls, tree: dict):
        return cls(from_tree(tree["ty"]))


class Bang(UnaryTyConstructor, Ty):
    def __init__(self, ty):
        if not isinstance(ty, biclosed.Ty):
            raise TypeError(messages.type_err(biclosed.Ty, ty))
        Ty.__init__(self, self)
        UnaryTyConstructor.__init__(self, ty)

    def project(self, n):
        return Project(n, self.ty)

    def __repr__(self):
        return "Bang({})".format(repr(self.ty))

    def __str__(self):
        return "!({})".format(self.ty)

    def __eq__(self, other):
        return isinstance(other, Bang) and self.ty == other.ty


class Nabla(UnaryTyConstructor, Ty):
    def __init__(self, ty: Ty):
        if not isinstance(ty, biclosed.Ty):
            raise TypeError(messages.type_err(biclosed.Ty, ty))
        Ty.__init__(self, self)
        UnaryTyConstructor.__init__(self, ty)

    def remove_nabla(self):
        return RemoveNabla(self.ty)

    def __repr__(self):
        return "Nabla({})".format(repr(self.ty))

    def __str__(self):
        return "∇({})".format(self.ty)

    def __eq__(self, other):
        return isinstance(other, Nabla) and self.ty == other.ty


class Project(Box):
    def __init__(self, n, ty):
        if n < 0 or not isinstance(n, int):
            raise ValueError("Number of copies `n` must be non-negative")
        self.ty = ty
        self.n = n
        dom = ty.bang()
        cod = Ty().tensor(*[ty] * n)
        super().__init__("Project({}, {})".format(n, ty), dom, cod)

    def __repr__(self):
        return "Project({}, {})".format(self.n, repr(self.ty))


class RemoveNabla(Box):
    def __init__(self, ty):
        self.ty = ty
        dom = ty.nabla()
        cod = ty
        super().__init__("RemoveNabla({})".format(ty), dom, cod)

    def __repr__(self):
        return "RemoveNabla({})".format(repr(self.ty))


class PermRight(Box):
    def __init__(self, nabla, right):
        if not isinstance(nabla, Nabla):
            raise TypeError(messages.type_err(Nabla, nabla))
        self.left = nabla
        self.right = right
        name = "PermRight({}, {})".format(nabla, right)
        dom = nabla @ right
        cod = right @ nabla
        super().__init__(name, dom, cod)


class PermLeft(Box):
    def __init__(self, left, nabla):
        if not isinstance(nabla, Nabla):
            raise TypeError(messages.type_err(Nabla, nabla))
        self.left = left
        self.right = nabla
        name = "PermLeft({}, {})".format(left, nabla)
        dom = left @ nabla
        cod = nabla @ left
        super().__init__(name, dom, cod)


class Functor(biclosed.Functor):
    """
    Functors into SLLM biclosed monoidal categories.

    Examples
    --------
    >>> from discopy import rigid
    >>> x, y = Ty('x'), Ty('y')
    >>> F = Functor(
    ...     ob={x: x, y: y}, ar={},
    ...     ob_factory=rigid.Ty,
    ...     ar_factory=rigid.Diagram)
    >>> print(F(y >> x << y))
    y.r @ x @ y.l
    >>> assert F((y >> x) << y) == F(y >> (x << y))
    """

    def __init__(self, ob, ar, ob_factory=Ty, ar_factory=Diagram):
        super().__init__(ob, ar, ob_factory, ar_factory)

    def __call__(self, diagram):
        if isinstance(diagram, Bang):
            return self.ob_factory.bang(self(diagram.ty))
        if isinstance(diagram, Nabla):
            return self.ob_factory.nabla(self(diagram.ty))
        if isinstance(diagram, Project):
            return self.ar_factory.project(diagram.n, self(diagram.ty))
        if isinstance(diagram, RemoveNabla):
            return self.ar_factory.remove_nabla(self(diagram.ty))
        if isinstance(diagram, PermLeft):
            return self.ar_factory.swap(self(diagram.left), self(diagram.right))
        if isinstance(diagram, PermRight):
            return self.ar_factory.swap(self(diagram.left), self(diagram.right))

        if isinstance(diagram, Ty) and len(diagram) > 1:
            return self.ob_factory.tensor(
                *[self(diagram[i : i + 1]) for i in range(len(diagram))]
            )
        return super().__call__(diagram)