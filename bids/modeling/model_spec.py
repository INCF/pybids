from abc import ABCMeta, abstractmethod

import pandas as pd
import numpy as np
from formulaic import model_matrix

from bids.utils import convert_JSON


def create_model_spec(df, model, *args, **kwargs):
    """Create and return a instance of the appropriate ModelSpec subclass.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame containing predictor data
    model : dict
        Dictionary containing the BIDS-StatsModels model information
    args, kwargs:
        Optional positional and keyword arguments to pass onto to subclass
        initializer.
    """
    kind = model.get('type', 'glm').lower()
    SpecCls = {
        'glm': GLMMSpec
    }[kind]
    return SpecCls.from_df(df, model, *args, **kwargs)


class ModelSpec(metaclass=ABCMeta):
    """Base class for all ModelSpec classes."""
    @abstractmethod
    def from_df(self, df, model, metadata=None):
        """Initialize from a pandas DataFrame."""
        pass


class GLMMSpec(ModelSpec):
    """Generalized Linear Mixed Model specification.
s
    Parameters
    ----------
    terms : list of Term
        A list of Term instances to include in the GLMMSpec instance.
    X: pd.DataFrame
        A pandas DataFrame containing the fixed effect design matrix
        (i.e., the X matrix in the typical mixed effect formulation). Each
        column will be internally converted to a separate Term instance.
    Z: pd.DataFrame
        A pandas DataFrame containing the random effect/grouping matrix
        (i.e., the Z matrix in the typical mixed effect formulation). Columns
        that share variance components are identified by the groups argument.
    groups: NDArray
        A binary 2d array with dimension k x v, where k is the number of
        columns in Z and v is the number of distinct variance components. A
        value of 1 indicates that the i'th of k rows is a level in the j's of
        v variance components. If Z is passed and groups is None, it is assumed
        that all columns in Z share the same single variance.
    sigma: NDArray
        An optional 2d array giving the covariance matrix for the variance
        components defined in the groups argument. Has dimension v x v, where v
        is the number of columns in groups.
    family: str
        The name of the family to use for the error distribution. By default,
        gaussian.
    link: str
        The name of the link function to use. Default depends on family. In the
        case of a gaussian (default family), an identity link is used.
    priors: dict
        Optional specification of default priors to use for new terms.
    """
    def __init__(self, terms=None, X=None, Z=None, groups=None, sigma=None,
                 family=None, link=None, priors=None):
        self.terms = {}
        self.family = family
        self.link = link
        self.sigma = sigma

        if priors is not None:
            self.set_priors(priors)

        if terms is not None:
            for t in terms:
                self.add_term(t)
        if X is not None:
            self.build_fixed_terms(X)
        if Z is not None:
            self.build_variance_components(Z, groups, sigma)

    def set_priors(self, fixed=None, random=None):
        raise NotImplementedError("Custom prior use hasn't been implemented yet.")

    def build_fixed_terms(self, X):
        """Build one or more fixed terms from the columns of a pandas DF.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas DataFrame containing variables to convert to Term
            instances. Each column is converted to a different (fixed) Term,
            with the name taken from the column name.
        """
        for col in X.columns:
            data = X.loc[:, col].values
            cat = data.dtype.name in ('str', 'category', 'object')
            # TODO: get default prior
            t = Term(col, data, categorical=cat)
            self.add_term(t)

    def build_variance_components(self, Z, groups=None, sigma=None, names=None):
        """Build one or more variance components from the columns of a binary
        grouping matrix and variance specification.

        Arguments:
            Z (DataFrame, NDArray): A binary 2D array or pandas DataFrame. Each
                column represents a column/predictor, each row represents an
                observation.
            groups (2DArray): A 2D binary array that maps the columns of Z
                onto variance components. Has dimension n_rows(Z) x k,
                where k is the number of distinct variance components. If None,
                a single group over all columns of Z is assumed.
            sigma (2DArray): A k x k 2D covariance matrix specifying the
                covariances between variance components. Currently unused.
            names (list): Optional list specifying the names of the groups. 
        """

        if sigma is not None:
            raise NotImplementedError("Covariance specification is currently"
                                      " not supported.")

        if groups is None:
            groups = np.ones((Z.shape[1], 1))
        n_grps = groups.shape[1]

        if names is None:
            names = getattr(groups, 'columns',
                            ['VC{}'.format(i) for i in range(n_grps)])

        # Work with array instead of DF
        if hasattr(groups, 'values'):
            groups = groups.values

        for i in range(n_grps):
            z_grp = Z[:, groups[:, i].astype(bool)]
            # TODO: select default prior
            vc = VarComp(names[i], z_grp)
            self.add_term(vc)

    def add_term(self, term):
        """Add a new Term to the instance.

        Parameters
        ----------
        term : Term
            A Term instance to add to the current instance.
        """
        if term.name in self.terms:
            raise ValueError("Term with name {} already exists!"
                             .format(term.name))
        self.terms[term.name] = term

    @property
    def X(self):
        """Return X design matrix (i.e., fixed component of model)."""
        if not self.fixed_terms:
            return None
        names, cols = zip(*[(c.name, c.values) for c in self.fixed_terms])
        return pd.DataFrame(np.c_[cols], columns=names)
        
    @property
    def Z(self):
        """Return Z design matrix (i.e., random effects/variance components).
        """
        if not self.variance_components:
            return None
        names, cols = [], []
        for c in self.variance_components:
            cols.append(c.values)
            names.extend(['{}.{}'.format(c.name, i)
                          for i in range(c.values.shape[1])])
        return pd.DataFrame(np.concatenate(cols, axis=1), columns=names)

    @property
    def fixed_terms(self):
        """Return a list of all available fixed effects."""
        return [t for t in self.terms.values() if not isinstance(t, VarComp)]

    @property
    def variance_components(self):
        """Return a list of all available variance components."""
        return [t for t in self.terms.values() if isinstance(t, VarComp)]

    @classmethod
    def from_df(cls, df, model, metadata=None, formula=None):
        """ Initialize a GLMMSpec instance from a BIDSVariableCollection and
        a BIDS-StatsModels JSON spec.

        Parameters
        ----------
        df : DataFrame
            A pandas DataFrame containing predictor information (i.e., the
            fixed component of the design matrix).
        model : dict
            The "Model" section from a BIDS-StatsModel specification.
        metadata: DataFrame
            Optional DataFrame containing additional columns that are not part
            of the design matrix but may have downstream informational use
            and/or contain variables needed to define random effects. Rows must
            map 1-to-1 with those in `df`.
        formula: str
            Optional Wilkinson (R-style) formula specifying the fixed (X) part
            of the design matrix. All variables referenced in the formula must
            be present as columns in `df`. Output names will follow the
            conventions specified in the `formulaic` documentation. Note that
            only the right-hand part of the formula should be passed (i.e.,
            pass "X1 * X2", not "y ~ X1 * X2"). If provided, willl take
            precedence over any formula found in the `model`.

        Returns
        -------
        A GLMMSpec instance.
        """

        kwargs = {}

        # Fixed terms
        model = convert_JSON(model)

        formula = formula or model.get('formula')
        if formula is not None:
            df = model_matrix(formula, df)

        kwargs['X'] = df

        # Variance components
        vcs = model.get('variance_components', [])
        Z_list = []

        if vcs:

            # VCs can be defined by variables in either the fixed predictor
            # DF or the supplementary metadata DF, so concatenate them.
            all_vars = [df, metadata] if metadata is not None else [df]
            all_vars = pd.concat(all_vars, axis=1)

            for vc in vcs:
                # Levels can either be defined by the levels of a single
                # categorical ("LevelsFrom") or by a set of binary variables.
                if 'levels_from' in vc:
                    data = all_vars[vc['levels_from']].values
                    Z_list.append(pd.get_dummies(data).values)
                else:
                    df = all_vars.loc[:, vc['levels']]
                    Z_list.append(df.values)

            Z = np.concatenate(Z_list, axis=1)
            groups = np.zeros((Z.shape[1], len(Z_list)))
            c = 0
            for i, vc in enumerate(Z_list):
                n = vc.shape[1]
                groups[c:(c+n), i] = 1
                c += n
            groups = pd.DataFrame(groups, columns=[vc['name'] for vc in vcs])

            kwargs['Z'] = Z
            kwargs['groups'] = groups

        error = model.get('error')
        if error:
            kwargs['family'] = error.get('family')
            kwargs['link'] = error.get('link')

        return GLMMSpec(**kwargs)


class Term(object):
    """Represents a model term.

    Parameters
    ----------
    name : str
        The name of the term.
    values : iterable
        A 1d array or other iterable containing the predictor values.
    categorical : bool
        Indicates whether or not the Term represents a categorical variable.
    prior : dict
        Optional specification of the prior distribution for the Term.
    metadata : dict
        Arbitrary metadata to store internally.
    """
    def __init__(self, name, values, categorical=False, prior=None,
                 metadata=None):
        self.name = name
        self.values = values
        self.categorical = categorical
        self.prior = prior
        self.metadata = metadata or {}


class VarComp(Term):
    """Represents a variance component/random effect.

    Parameters
    ----------
    name : str
        The name of the variance component.
    values : iterable
        A 2d binary array identifying the observations that belong to the
        levels of the variance component. Has dimension n x k, where n is the
        number of observed rows in the dataset and k is the number of levels
        in the factor.
    prior : dict
        Optional specification of the prior distribution for the VarComp.
    metadata : dict
        Arbitrary metadata to store internally.
    """
    def __init__(self, name, values, prior=None, metadata=None):
        super(VarComp, self).__init__(name, values, categorical=True,
                                      prior=prior, metadata=metadata)
        self.index_vec = self.dummies_to_vec(values)

    @staticmethod
    def dummies_to_vec(dummies):
        """Convert dummy-coded columns to a single integer index.

        Parameters
        ----------
        dummies : NDArray
            2d binary array to recode as a single vector.

        Notes
        -----
        Used for the sake of computational efficiency (i.e., to avoid lots of
        large matrix multiplications in the backends), invert the dummy-coding
        process and represent full-rank dummies as a vector of indices into the
        coefficients.
        """
        vec = np.zeros(len(dummies), dtype=int)
        for i in range(dummies.shape[1]):
            vec[(dummies[:, i] == 1)] = i + 1
        return vec


class Prior(object):
    '''Abstract specification of a term prior.

    Parameters
    ----------
    name : str
        Name of prior distribution (e.g., Normal, Bernoulli, etc.)
    kwargs: dict
        Optional keywords specifying the parameters of the named distribution.

    Notes
    -----
    At present there's no controlled vocabulary of supported prior names and
    arguments, but users implementing new Bayesian estimators are encouraged to
    use the names used in PyMC3 (e.g., 'Normal', parameterized with mu and
    sd arguments).
    '''
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs
