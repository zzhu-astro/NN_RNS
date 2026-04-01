import numpy as np
import warnings
from scipy.interpolate import UnivariateSpline
from . import units

class EoSTable:
    """
    EOS table:
      - read_data(source, fmt=...)
      - p_from_nb(nb): interpolate P(nb) (log-log interpolation)
    """

    def __init__(self, source, format="RNS"):
        self.lognb, self.logp, self.loge = self.read_data(source, format=format)
        self.nb_p_interp = UnivariateSpline(self.lognb, self.logp, s=0, k=3)
        self.p_nb_interp = UnivariateSpline(self.logp, self.lognb, s=0, k=3)
        self.p_e_interp  = UnivariateSpline(self.logp, self.loge, s=0, k=3)

    def read_data(self, source, format="RNS"):
        """
        Read EOS data from a filename or a numpy array.

        Parameters
        ----------
        source : str | np.ndarray
            filename or array with shape (N, M)
        format : "RNS" | "lorene"
            only used to choose default columns if you want

        Returns
        -------
        EOSTable
        """
        if format.lower() == "rns":
            with open(source, "r") as f:
                lines = f.readlines()

            # find first non-empty non-comment line
            first_idx = None
            for i, line in enumerate(lines):
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#") or s.startswith("%") or s.startswith("!"):
                    continue
                first_idx = i
                first_line = s
                break

            if first_idx is None:
                raise ValueError(f"Empty EOS file: {source}")

            # check if first data line is a single integer
            tokens = first_line.split()
            has_length_header = False
            if len(tokens) == 1:
                try:
                    int(tokens[0])
                    has_length_header = True
                except ValueError:
                    pass

            # create a text buffer for np.loadtxt
            if has_length_header:
                table_lines = lines[first_idx + 1 :]
            else:
                table_lines = lines[first_idx :]

            # np.loadtxt can read from list of strings directly
            data = np.loadtxt(table_lines)
            lognb = np.log(data[:,3])
            logp  = np.log(data[:,1])
            loge  = np.log(data[:,0])

            # sanity check
            for name, arr in (("lognb", lognb), ("logp", logp), ("loge", loge)):
                if not np.all(np.isfinite(arr)):
                    bad = np.where(~np.isfinite(arr))[0][0]
                    raise ValueError(f"{name} contains non-finite values at index {bad}: {arr[bad]}")
            
                # check if strictly increasing
                d = np.diff(arr)
                bad = np.where(d <= 0)[0]
                if bad.size:
                    i = int(bad[0])
                    raise ValueError(f"{name} must be strictly increasing; fails at {i}->{i+1}: {arr[i]} -> {arr[i+1]}")

            return lognb, logp, loge

        elif format.lower() == "lorene":
            raise ValueError("Lorene table reading not implemented yet")
        else:
            raise ValueError("format of EoS must be 'rns' or 'lorene'")

    def p_from_nb(self, nb_new):
        """
        Interpolate pressure P(nb) using log-log interpolation.
        nb can be scalar or array, and unit of nb is fm^-3
        """
        nb_new = np.asarray(nb_new, dtype=float)
        lognb_new = np.log(nb_new)

        assert np.nanmin(lognb_new) >= self.lognb[0], (
            f"nb of NN below table range: min(log(nb))={np.nanmin(lognb_new)} < lognb_min={self.lognb[0]}"
        )

        lognb_max = self.lognb[-1]

        # interpolate for in-range points
        logp_new = np.empty_like(lognb_new, dtype=float)
        self.in_range = lognb_new <= lognb_max
        logp_new[self.in_range] = self.nb_p_interp(lognb_new[self.in_range])

        # linear extrapolation for out-of-range points (log-log space)
        out_range = ~self.in_range
        if np.any(out_range):
            warnings.warn(
                "nb of NN exceeds EOS table max; using linear extrapolation.",
                RuntimeWarning,
                stacklevel=2,
            )
            # slope from last two tabulated points in (lognb, logp)
            dlognb = self.lognb[-1] - self.lognb[-2]
            if dlognb == 0:
                slope = 0.0
            else:
                slope = (self.logp[-1] - self.logp[-2]) / dlognb
            logp_new[out_range] = self.logp[-1] + slope * (lognb_new[out_range] - self.lognb[-1])

        return np.exp(logp_new)
    


    def nb_from_p(self, p_new):
        """
        Interpolate number density nb(P) using log-log interpolation.
        p can be scalar or array, and unit of p is CGS
        """
        p_new = np.asarray(p_new, dtype=float)
        logp_new = np.log(p_new)

        # interpolate for in-range points
        lognb_new = np.empty_like(logp_new, dtype=float)
        lognb_new[:] = self.p_nb_interp(logp_new[:])

        return np.exp(lognb_new)
    

    def e_from_p(self, p_new):
        """
        Interpolate energy density e(P) using log-log interpolation.
        p can be scalar or array, and unit of p is CGS
        """
        p_new = np.asarray(p_new, dtype=float)
        logp_new = np.log(p_new)

        # interpolate for in-range points
        loge_new = np.empty_like(logp_new, dtype=float)
        loge_new[:] = self.p_e_interp(logp_new[:])

        return np.exp(loge_new)
