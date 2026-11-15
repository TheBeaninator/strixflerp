"""Module closures: what a set of root modules pulls in through `requires`, under a recipe's interface bindings."""
from .catalog import overlap_class, split_tag


def satisfies(module, iface):
    """The interface of `module` that fills a requirement for `iface`: the same full name, or - when the
    requirement carries no tag - a tagged name of the same overlap class (a provider whose numeric contract
    differs; the recipe chose it knowingly). None when the module provides neither."""
    if iface in module.interfaces:
        return iface
    if split_tag(iface)[1] is None:
        return next((p for p in module.interfaces if overlap_class(p) == iface), None)
    return None


def direct_deps(cat, module, bindings, errors, used=None, changed=None, overlays=None):
    """Module ids `module` requires. bindings: interface -> module id (a recipe's [modules]); None = no recipe,
    where an interface resolves only if exactly one module in the catalogue provides it. overlays: origin module id ->
    derived module id (a recipe's [overlay]): an overlaid module also requires what its derived module requires
    (the derived sources compile in its target), except the origin itself."""
    out = []
    reqs = list(module.requires)
    if overlays and module.id in overlays and overlays[module.id] in cat.by_id:
        reqs += [r for r in cat.by_id[overlays[module.id]].requires if r.get("module") != module.id]
    for r in reqs:
        if "module" in r:
            dep = r["module"]
            if dep not in cat.by_id:
                errors.append(f"{module.id}: requires module {dep}, which is not in the catalogue")
                continue
        else:
            iface = r.get("interface")
            if bindings is None:
                provs = cat.providers.get(iface, [])
                if len(provs) != 1:
                    what = "no provider" if not provs else "providers " + ", ".join(p.id for p in provs)
                    errors.append(f"{module.id}: interface {iface} has {what}; a recipe must choose")
                    continue
                dep = provs[0].id
            else:
                dep = bindings.get(iface)
                if dep is None:
                    errors.append(f"{module.id}: interface {iface} is unfilled (no entry in [modules])")
                    continue
                if dep not in cat.by_id:
                    errors.append(f"[modules] binds {iface} to {dep}, which is not in the catalogue")
                    continue
                got = satisfies(cat.by_id[dep], iface)
                if got is None:
                    errors.append(f"[modules] binds {iface} to {dep}, which does not provide it")
                    continue
                if got != iface and changed is not None and (iface, got, dep) not in changed:
                    changed.append((iface, got, dep))
                if used is not None:
                    used.add(iface)
        if dep not in out and dep != module.id:
            out.append(dep)
    return out


def closure(cat, roots, bindings, errors, used=None, changed=None, overlays=None):
    """Ids of the roots and everything they require, each after its dependencies where the graph allows
    (a dependency cycle is reported and broken)."""
    order, state = [], {}

    def visit(mid, trail):
        if state.get(mid) == "done":
            return
        if state.get(mid) == "open":
            cyc = trail[trail.index(mid):] + [mid]
            errors.append("dependency cycle: " + " -> ".join(cyc))
            return
        state[mid] = "open"
        for dep in direct_deps(cat, cat.by_id[mid], bindings, errors, used, changed, overlays):
            visit(dep, trail + [mid])
        state[mid] = "done"
        order.append(mid)

    for root in roots:
        if root not in cat.by_id:
            errors.append(f"module {root} is not in the catalogue")
            continue
        visit(root, [])
    return order


class Assembly:
    """A recipe resolved against the catalogue."""

    def __init__(self, cat, recipe):
        self.cat, self.recipe = cat, recipe
        self.errors, self.warnings = [], []
        used = set()
        # [overlay]: derived module id -> origin module id. The derived module's sources compile in the origin's
        # target in place of the origin's files at the same origin-relative path, its include root ahead of the
        # origin's wherever the origin's is on a include path (SPEC.md section 5, derived modules).
        self.overlays = {}                     # origin module id -> derived module id
        ov = recipe.data.get("overlay", {})
        for did, oid in (ov.items() if isinstance(ov, dict) else ()):
            if did not in cat.by_id or oid not in cat.by_id:
                self.errors.append(f"[overlay] {did} = {oid}: both must be modules of the catalogue")
            elif not any(r.get("module") == oid for r in cat.by_id[did].requires):
                self.errors.append(f"[overlay] {did} must require {oid}, the module it overlays")
            elif oid in self.overlays:
                self.errors.append(f"[overlay] {oid} is overlaid by both {self.overlays[oid]} and {did}")
            else:
                self.overlays[oid] = did
        self.contract_changes = []             # (required interface, provided tagged interface, module id)
        changed = self.contract_changes
        self.binaries = {}                     # name -> (main id, entry origin path, closure ids)
        for name, spec in recipe.binaries.items():
            if not isinstance(spec, dict) or not isinstance(spec.get("main"), str):
                self.errors.append(f"[binaries] {name}: needs main = \"<module id>\"")
                continue
            main = spec["main"]
            if main not in cat.by_id:
                self.errors.append(f"[binaries] {name}: main module {main} is not in the catalogue")
                continue
            entry = spec.get("entry")
            sources = cat.by_id[main].sources
            if entry is not None and "src/" + str(entry) not in sources:
                self.errors.append(f"[binaries] {name}: entry {entry} is not a source of {main}")
            if not sources:
                self.errors.append(f"[binaries] {name}: main module {main} has no sources")
            self.binaries[name] = (main, entry, closure(cat, [main], recipe.bindings, self.errors, used, changed,
                                                        self.overlays))
        mains = {b[0] for b in self.binaries.values()}
        for name, (main, _, ids) in self.binaries.items():
            for other in mains & set(ids) - {main}:
                self.errors.append(f"[binaries] {name}: its closure contains {other}, another binary's main")
        self.include = [i for i in recipe.include if isinstance(i, str)]
        extra = closure(cat, self.include, recipe.bindings, self.errors, used, changed, self.overlays)
        self.modules = []
        for ids in [b[2] for b in self.binaries.values()] + [extra]:
            self.modules += [i for i in ids if i not in self.modules]
        # A cycle between modules is a fact of some origins (headers of two modules including each other). The
        # closure is the same set either way and objects link in any order, so it is reported, not refused.
        for oid, did in self.overlays.items():
            if oid not in self.modules:
                self.errors.append(f"[overlay] {did} overlays {oid}, which is not in the assembly")
            elif did not in self.modules:
                self.modules.append(did)       # recorded; its sources compile in the origin's target
        self.warnings += list(dict.fromkeys(e for e in self.errors if e.startswith("dependency cycle")))
        self.errors = list(dict.fromkeys(e for e in self.errors if not e.startswith("dependency cycle")))
        for iface in recipe.bindings:
            if iface not in used:
                self.warnings.append(f"[modules] {iface} is bound but nothing in the assembly requires it")
        # Adapters: module id -> symbols an adapter of the assembly takes over from it (renamed at compile time).
        self.renames, self.adapters = {}, {}
        for mid in self.modules:
            target, symbols = cat.by_id[mid].overrides
            if target is None:
                continue
            if target in self.adapters:
                self.errors.append(f"{mid} and {self.adapters[target]} both adapt {target}")
                continue
            self.adapters[target] = mid
            self.renames[target] = list(symbols)
        seen = {}
        for mid in self.modules:
            for iface in cat.by_id[mid].interfaces:
                # an adapter and the module whose API it re-provides may name the same interface: the adapter's
                # symbols replace the module's, so nothing is defined twice
                other = seen.get(iface)
                if other is not None and self.adapters.get(other) != mid and self.adapters.get(mid) != other:
                    self.errors.append(f"interface {iface} is provided twice in the assembly: {other} and {mid}")
                seen[iface] = mid

    def deps(self, mid):
        """Transitive dependency ids of one module, dependencies first, without the module itself."""
        return [i for i in closure(self.cat, [mid], self.recipe.bindings, [], overlays=self.overlays) if i != mid]
