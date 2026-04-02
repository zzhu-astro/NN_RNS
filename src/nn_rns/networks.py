from .eos import EoSTable
from .NN.nn import NN_models
import numpy as np
from scipy.optimize import newton, bisect
from scipy.interpolate import UnivariateSpline
import warnings


class RNSNetworks:
    """
    networks:
      - load RNS networks and evaluate
    """

    def __init__(self):
        # load models
        self.nn_models = NN_models()
        self.obs_names = self.nn_models.obs_names
        self.r_state_names = self.nn_models.r_state_names
        self.static_obs_names = self.nn_models.static_obs_names
        self.log_obs = self.nn_models.log_obs

    def rns_eval(self, eos_table):
        self.nn_models.nn_eval(eos_table)

        # Keep EOS table for later central-variable conversions.
        self.eos_table = eos_table
        self.eos_mask = self.nn_models.eos_mask

        self.nn_rns_static = self.nn_models.nn_rns_static
        self.nn_rns_kepler = self.nn_models.nn_rns_kepler
        self.nn_rns_rotate = self.nn_models.nn_rns_rotate
        self.press_c = self.nn_models.press_c
        self.energy_c = self.nn_models.energy_c
        self.nb_c = self.nn_models.nb_c

    def compute_m_max(self, rot_input, rot_input_type="Omega"):
        """
        Compute max-mass quantities from M(r_ratio, e_c) slices.

        Parameters
        ----------
        rot_input : array-like
            Target rotation inputs. Interpreted as r_ratio values if
            rot_input_type="r_ratio", or as Omega values if
            rot_input_type="Omega".
        rot_input_type : {"r_ratio", "Omega"}
            Meaning of rot_input.

        Returns
        -------

        """
        if not hasattr(self, "energy_c"):
            raise RuntimeError("Run rns_eval(EoSTable) before compute_m_max.")

        rot_kind = str(rot_input_type)
        if rot_kind not in {"r_ratio", "Omega"}:
            warnings.warn(
                "Invalid rot_input_type; using default 'Omega'.",
                RuntimeWarning,
                stacklevel=2,
            )
            rot_kind = "Omega"

        rot_vals = np.asarray(rot_input, dtype=float)
        if rot_vals.ndim == 0:
            rot_vals = rot_vals.reshape(1)

        all_obs = self._build_all_observables()
        M_2darr = all_obs[:, :, 0]
        R_2darr = all_obs[:, :, 2]
        Omega_2darr = all_obs[:, :, 3]
        rr_2darr = all_obs[:, :, 14]
        e_c = np.asarray(self.energy_c, dtype=float)

        n_ratio = M_2darr.shape[0]
        mmax_seq = np.full((n_ratio,), np.nan, dtype=float)
        R_max_seq = np.full((n_ratio,), np.nan, dtype=float)
        omgmax_seq = np.full((n_ratio,), np.nan, dtype=float)
        ecmax_seq = np.full((n_ratio,), np.nan, dtype=float)
        rrmax_seq = np.full((n_ratio,), np.nan, dtype=float)

        for i_ratio in range(n_ratio):
            M_arr = np.asarray(M_2darr[i_ratio, :], dtype=float)
            R_arr = np.asarray(R_2darr[i_ratio, :], dtype=float)
            Omega_arr = np.asarray(Omega_2darr[i_ratio, :], dtype=float)
            rr_arr = np.asarray(rr_2darr[i_ratio, :], dtype=float)

            mask = ( (M_arr > 0.5) & (R_arr < 30.0) )

            ec_valid = e_c[mask]
            M_valid = M_arr[mask]
            R_valid = R_arr[mask]
            Omega_valid = Omega_arr[mask]

            # First local peak index i_max in M_valid.
            dM = np.diff(M_valid)
            cand = np.where((dM[:-1] > 0.0) & (dM[1:] <= 0.0))[0] + 1
            if cand.size > 0:
                i_max = int(cand[0])
            else:
                raise RuntimeError(f"No valid maximum mass found for r_ratio index {i_ratio}.")

            # 3-point interpolation with (i_max-1, i_max, i_max+1).
            if i_max == 0 or i_max == (M_valid.size - 1):
                ec_max = ec_valid[i_max]
                m_max = M_valid[i_max]
                Omega_max = Omega_valid[i_max]
                R_max = R_valid[i_max]
                rrmax_seq[i_ratio] = rr_arr[mask][i_max]
                mmax_seq[i_ratio] = m_max
                R_max_seq[i_ratio] = R_max
                omgmax_seq[i_ratio] = Omega_max
                ecmax_seq[i_ratio] = ec_max
                continue

            x3 = ec_valid[i_max - 1 : i_max + 2]
            y3 = M_valid[i_max - 1 : i_max + 2]
            yO3 = Omega_valid[i_max - 1 : i_max + 2]
            a, b, c = np.polyfit(x3, y3, 2)

            ec_max = float(-b / (2.0 * a))
            ec_max = float(np.clip(ec_max, np.min(x3), np.max(x3)))
            m_max = float(a * ec_max * ec_max + b * ec_max + c)

            Omega_spline = UnivariateSpline(ec_valid, Omega_valid, s=0, k=3)
            Omega_max = Omega_spline(ec_max)

            R_spline = UnivariateSpline(ec_valid, R_valid, s=0, k=3)
            R_max = R_spline(ec_max)

            if i_ratio==0:
                rr_spline = UnivariateSpline(ec_valid, rr_arr[mask], s=0, k=3)
                rrmax_seq[i_ratio] = rr_spline(ec_max)

            mmax_seq[i_ratio] = m_max
            R_max_seq[i_ratio] = R_max
            omgmax_seq[i_ratio] = Omega_max
            ecmax_seq[i_ratio] = ec_max
            rrmax_seq[i_ratio] = rr_arr[mask][i_max]

        # return mmax_seq, R_max_seq, omgmax_seq, ecmax_seq, rrmax_seq
        self.mmax_seq = mmax_seq
        self.R_max_seq = R_max_seq
        self.omgmax_seq = omgmax_seq
        self.ecmax_seq = ecmax_seq
        self.rrmax_seq = rrmax_seq

        # Check peak and monotonicity of mmax_seq and omgmax_seq.
        rr_kep = rrmax_seq[0]

        valid_idx_mmax   = np.where(mmax_seq <= mmax_seq[0])[0]
        valid_idx_omgmax = np.where(mmax_seq <= mmax_seq[0])[0]
        mmax_seq_valid   = mmax_seq[valid_idx_mmax]
        omgmax_seq_valid = omgmax_seq[valid_idx_omgmax]
        rrmax_seq_valid  = rrmax_seq[valid_idx_mmax]

        left_idx = np.where(rrmax_seq_valid[1:] < rr_kep)[0] + 1
        right_idx = np.where(rrmax_seq_valid[1:] > rr_kep)[0] + 1

        left_array = omgmax_seq_valid[left_idx]
        right_array = omgmax_seq_valid[right_idx]

        left_mono_idx_s  = self._longest_strictmono_indices(left_array, order="increasing")
        right_mono_idx_s = self._longest_strictmono_indices(right_array, order="decreasing")


        if len(left_mono_idx_s)*len(right_mono_idx_s) != 1:
            best_idx   = None
            best_score = np.inf
            idx_max_t = None
            for nl in range(len(left_mono_idx_s)):
                for nr in range(len(right_mono_idx_s)):
                    mono_idx_tmp = np.concatenate([left_idx[left_mono_idx_s[nl]], np.array([0]), right_idx[right_mono_idx_s[nr]] ])
                    score = self._smoothness_score(rrmax_seq_valid[mono_idx_tmp], omgmax_seq_valid[mono_idx_tmp])
                    if score < best_score:
                        best_score = score
                        best_idx = mono_idx_tmp
                        idx_max_t = len(left_mono_idx_s[nl])

            valid_mono_idx = best_idx
            idx_max = idx_max_t
        else:
            valid_mono_idx = np.concatenate([left_idx[left_mono_idx_s[0]], np.array([0]), right_idx[right_mono_idx_s[0]] ])
            idx_max = len(left_mono_idx_s)

        if valid_mono_idx.size < 4:
            raise RuntimeError("Not enough valid monotonic points for interpolation.")
        
        # interpolation of mmax
        rr_mono = rrmax_seq_valid[valid_mono_idx]
        omg_mono = omgmax_seq_valid[valid_mono_idx]
        mmax_mono = mmax_seq_valid[valid_mono_idx]

        # linear spline interpolation to remain the peak
        rr_omg_spline = UnivariateSpline(rr_mono, mmax_mono, s=0, k=1)
        rr_mmax_spline = UnivariateSpline(rr_mono, omg_mono, s=0, k=1)

        if rot_kind == "r_ratio":
            rr_target = np.asarray(rot_vals, dtype=float).copy()
            rr_min = float(np.min(rr_mono))
            rr_max = float(np.max(rr_mono))
            out_mask = (rr_target < rr_min) | (rr_target > rr_max)
            if np.any(out_mask):
                warnings.warn(
                    "Some rot_input r_ratio values are out of rr_mono bounds; clipped to min/max.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                rr_target = np.clip(rr_target, rr_min, rr_max)

            mmax_interp = rr_omg_spline(rr_target)
            omg_interp = rr_mmax_spline(rr_target)
            return mmax_interp, omg_interp

        if rot_kind == "Omega":
            omg_target = np.asarray(rot_vals, dtype=float).copy()
            omg_min = float(np.min(omg_mono))
            omg_max = float(np.max(omg_mono))
            out_mask = (omg_target < omg_min) | (omg_target > omg_max)
            if np.any(out_mask):
                warnings.warn(
                    "Some rot_input Omega values are out of omg_mono bounds; clipped to min/max.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                omg_target = np.clip(omg_target, omg_min, omg_max)

            rr_target = np.full_like(omg_target, np.nan, dtype=float)
            rr_min = rr_mono[idx_max]
            rr_max = 1.0
            for i, omg_t in enumerate(omg_target):
                x0 = float(rr_mono[np.argmin(np.abs(omg_mono - omg_t))])

                def _f(rr_val):
                    return float(rr_mmax_spline(rr_val) - omg_t)

                try:
                    f_lo = _f(rr_min)
                    f_hi = _f(rr_max)
                    if abs(f_lo) <= 1e-12:
                        rr_sol = rr_min
                    elif abs(f_hi) <= 1e-12:
                        rr_sol = rr_max
                    elif f_lo * f_hi < 0.0:
                        rr_sol = bisect(_f, rr_min, rr_max, xtol=1e-10, maxiter=100)
                    else:
                        rr_sol = x0
                except Exception:
                    rr_sol = x0
                rr_target[i] = float(np.clip(rr_sol, rr_min, rr_max))

            mmax_interp = rr_omg_spline(rr_target)
            return mmax_interp, rr_target
    # Longest strict monotonic subsequence indices.
    # If return_all=True, return all longest index sequences.
    def _longest_strictmono_indices(self, y, order="increasing", return_all=True):
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if order not in {"increasing", "decreasing"}:
            raise ValueError("order must be 'increasing' or 'decreasing'.")

        vals = y if order == "increasing" else -y
        n = vals.size
        if n == 0:
            return [] if return_all else np.array([], dtype=int)

        # O(n^2) DP with predecessor graph to preserve all optimal paths.
        dp_len = np.ones(n, dtype=int)
        prev_nodes = [[] for _ in range(n)]

        for i in range(n):
            best_len = 1
            preds = []
            for j in range(i):
                if vals[j] < vals[i]:
                    cand_len = int(dp_len[j] + 1)
                    if cand_len > best_len:
                        best_len = cand_len
                        preds = [j]
                    elif cand_len == best_len:
                        preds.append(j)
            dp_len[i] = best_len
            prev_nodes[i] = preds

        max_len = int(np.max(dp_len))
        end_nodes = np.where(dp_len == max_len)[0]

        all_paths = []

        def _dfs(node, rev_path):
            rev_path.append(int(node))
            if not prev_nodes[node]:
                all_paths.append(np.asarray(rev_path[::-1], dtype=int))
            else:
                for p in prev_nodes[node]:
                    _dfs(p, rev_path)
            rev_path.pop()

        for end in end_nodes:
            _dfs(int(end), [])

        # Deterministic order: lexicographic by index tuple.
        all_paths.sort(key=lambda arr: tuple(arr.tolist()))

        if return_all:
            return all_paths

        return all_paths[0] if all_paths else np.array([], dtype=int)




    def _smoothness_score(self, rr, omg, alpha=1.0, beta=0.05, eps=1e-12):
        rr = np.asarray(rr, dtype=float)
        omg = np.asarray(omg, dtype=float)

        if rr.size < 3:
            return np.inf  # need at least 3 points to assess curvature-like smoothness

        dx = np.diff(rr)
        dy = np.diff(omg)
        if np.any(dx <= eps):
            return np.inf  # invalid for strict monotonic rr mapping

        # First derivative d(omg)/d(rr)
        s = dy / dx
        if s.size < 2:
            return np.inf

        # Second derivative on nonuniform grid
        dx_mid = dx[:-1] + dx[1:]
        if np.any(dx_mid <= eps):
            return np.inf
        c = 2.0 * (s[1:] - s[:-1]) / dx_mid

        roughness = np.mean(c**2)

        # Optional spacing regularity
        dx_mean = np.mean(dx)
        spacing_penalty = np.mean((dx - dx_mean) ** 2)

        return float(alpha * roughness + beta * spacing_penalty)



    def compute_observables(
        self,
        rot_input,
        central_input,
        rot_input_type="Omega",
        central_input_type="e_c",
        spline_order=3
        ):
        """
        Calculate interpolated observables on a target (rotation, central-variable) grid.

        Parameters
        ----------
        rot_input : array-like
            Input rotation coordinate. Interpreted as r_ratio or Omega per
            rot_input_type.
        central_input : array-like
            Central variable values. Interpreted as nb_c, e_c, or p_c per
            central_input_type.
        rot_input_type : {"r_ratio", "Omega"}
            Selects the meaning of rot_input.
        central_input_type : {"nb_c", "e_c", "p_c"}
            Selects the meaning of central_input.
            The units: nb_c in fm^-3, e_c in g/cm^3 (CGS), p_c in dyn/cm^3 (CGS).
        r_ratio_index : int
            Column index of r_ratio in the observable vector.
        omega_index : int
            Column index of Omega in the observable vector.

        Returns
        -------
        np.ndarray
            Shape (n, m, len_obs) where n = len(rot_input),
            m = len(central_input converted to e_c), and len_obs is the
            observable-vector length.
        """
        if not hasattr(self, "energy_c"):
            raise RuntimeError("Run rns_eval(EoSTable) before calculate_observables.")

        rot_kind = str(rot_input_type)
        cen_kind = str(central_input_type)
        if rot_kind not in {"r_ratio", "Omega"}:
            warnings.warn(
                "Invalid rot_input_type; using default 'Omega'.",
                RuntimeWarning,
                stacklevel=2,
            )
            rot_kind = "Omega"
        if cen_kind not in {"nb_c", "e_c", "p_c"}:
            warnings.warn(
                "Invalid central_input_type; using default 'e_c'.",
                RuntimeWarning,
                stacklevel=2,
            )
            cen_kind = "e_c"

        rot_vals = np.asarray(rot_input, dtype=float)
        if rot_vals.ndim == 0:
            rot_vals = rot_vals.reshape(1)
        if rot_kind == "r_ratio":
            out_of_range = (rot_vals < 0.5) | (rot_vals > 1.0)
            if np.any(out_of_range):
                warnings.warn(
                    "Some r_ratio values are outside [0.5, 1.0]; clipping to boundaries.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                rot_vals = np.clip(rot_vals, 0.5, 1.0)

        central_vals = np.asarray(central_input, dtype=float)
        if central_vals.ndim == 0:
            central_vals = central_vals.reshape(1)

        e_c_target = self._to_energy_c(central_vals, cen_kind)
        if np.any(e_c_target <= 0):
            raise ValueError("All converted e_c values must be positive.")

        all_obs = self._build_all_observables()
        all_obs_ec = self._interpolate_obs_vs_log_energy(all_obs, e_c_target, spline_order=spline_order)


        n_rot = rot_vals.size
        n_ec = e_c_target.size
        len_obs = all_obs.shape[2]
        out = np.full((n_rot, n_ec, len_obs), np.nan, dtype=float)

        # Interpolated r_ratio of Kepler case, used as lower limit per e_c.
        rr_kepler    = all_obs_ec[0, :, 14]
        Omega_kepler = all_obs_ec[0, :, 3]
        for i_ec in range(n_ec):
            obs_slice = all_obs_ec[:, i_ec, :]
            rr_slice  = obs_slice[:, 14]
            omg_slice = obs_slice[:, 3]

            for i_rot, rot_val in enumerate(rot_vals):
                if rot_kind == "r_ratio":
                    rr_target = float(rot_val)
                else:
                    if float(rot_val) < float(Omega_kepler[i_ec]):
                        rr_target = self._solve_rr_from_omega(omg_slice, rr_slice, float(rot_val))
                    else:
                        rr_target = rr_kepler[i_ec]

                    # Enforce rr_target >= rr_kepler[i_ec]
                    if np.isfinite(rr_kepler[i_ec]) and rr_target < rr_kepler[i_ec]:
                        rr_target = float(rr_kepler[i_ec])

                out[i_rot, i_ec, :] = self._interp_obs_at_rr(obs_slice, rr_slice, rr_target)

        return out

    def _to_energy_c(self, central_vals, central_kind):
        if central_kind == "e_c":
            return np.asarray(central_vals, dtype=float)

        if not hasattr(self, "eos_table"):
            raise RuntimeError(
                "EOS table is not available. Run rns_eval(EoSTable) first."
            )

        if central_kind == "nb_c":
            p_vals = self.eos_table.p_from_nb(central_vals)
            return self.eos_table.e_from_p(p_vals)

        # central_kind == "p_c"
        return self.eos_table.e_from_p(central_vals)

    def _build_all_observables(self):
        n_ec = self.energy_c.shape[0]
        n_rot_cases = self.nn_rns_rotate.shape[0]
        len_obs = len(self.obs_names)
        n_cases = n_rot_cases + 2  # kepler + rotate cases + static

        all_obs = np.full((n_cases, n_ec, len_obs), np.nan, dtype=float)

        # Static case: directly available columns are 
        # [M, M_0, R, Omega_p, Z_p].
        # rotation case columns
        # [M, M_0, R, Omega, T/W, C*J/GM_s^2, I, Phi_2, h_plus, h_minus, Z_p, Z_b, Z_f, Omega_p]
        # kepler case columns
        # [M, M_0, R, Omega, T/W, C*J/GM_s^2, I, Phi_2, h_plus, h_minus, Z_p, Z_b, Z_f, Omega_p, r_ratio]
        static = self.nn_rns_static
        all_obs[11, :, 0] = static[:, 0]  # M
        all_obs[11, :, 1] = static[:, 1]  # M_0
        all_obs[11, :, 2] = static[:, 2]  # R
        all_obs[11, :, 3:6]   = 0.0       # Omega, T/W, C*J/GM_s^2
        all_obs[11, :, 8:10]  = 0.0       # h_plus, h_minus
        all_obs[11, :, 10] = static[:, 4] # Z_p
        all_obs[11, :, 11] = static[:, 4] # Z_b
        all_obs[11, :, 12] = static[:, 4] # Z_f
        all_obs[11, :, 13] = static[:, 3] # Omega_p
        all_obs[11, :, 14] = 1.0          # r_ratio

        # Rotation sequence.
        all_obs[1 : 1 + n_rot_cases, :, :14] = self.nn_rns_rotate
        r_ratio_grid = np.tile(np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95], dtype=float)[:, np.newaxis], (1, n_ec))
        all_obs[1 : 1 + n_rot_cases, :, 14] = r_ratio_grid

        # Kepler endpoint.
        all_obs[0, :, :13] = self.nn_rns_kepler[:, :13]
        all_obs[0, :, 13]  = self.nn_rns_kepler[:, 3]   # Omega_p
        all_obs[0, :, 14]  = self.nn_rns_kepler[:, 13]

        return all_obs

    def _interpolate_obs_vs_log_energy(self, all_obs, e_c_target, spline_order=3):
        loge_src = np.log(np.asarray(self.energy_c, dtype=float)[self.eos_mask])
        loge_tgt = np.log(np.asarray(e_c_target, dtype=float))

        n_rotate, _, len_obs = all_obs.shape
        n_tgt = loge_tgt.size
        out = np.full((n_rotate, n_tgt, len_obs), np.nan, dtype=float)

        self.ec_obs_interp_dict    = {}                    
        for i_rotate, r_state in enumerate(self.r_state_names):
            self.ec_obs_interp_dict[r_state] = {}
            for i_obs, obs in enumerate(self.obs_names):

                if r_state == "static" and obs not in self.static_obs_names:
                    if obs in ['Omega', 'T/W', 'J', 'h_plus', 'h_minus']:
                        out[i_rotate, :, i_obs] = 0.0
                    continue
                if r_state != "kepler" and obs == "r_ratio":
                    continue
                if r_state == "kepler" and obs == "Omega_p":
                    continue

                y = all_obs[i_rotate, :, i_obs][self.eos_mask]
                
                self.ec_obs_interp_dict[r_state][obs] = UnivariateSpline(loge_src, y, s=0, k=spline_order)
                out[i_rotate, :, i_obs] = self.ec_obs_interp_dict[r_state][obs](loge_tgt)

                if obs in self.log_obs:
                    out[i_rotate, :, i_obs] = np.exp(out[i_rotate, :, i_obs])

        r_ratio_grid = np.tile(np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95], dtype=float)[:, np.newaxis], (1, n_tgt))
        out[1:11, :, 14] = r_ratio_grid           # rotation cases
        out[11, :, 14]   = 1.0                    # static case

        return out

    def _interp_obs_at_rr(self, obs_slice, rr_slice, rr_target):
        len_obs = obs_slice.shape[1]
        out = np.full((len_obs,), np.nan, dtype=float)

        rr_mask = np.isfinite(rr_slice)
        if np.count_nonzero(rr_mask) < 2:
            return out

        for i_obs in range(len_obs):
            y = obs_slice[:, i_obs]
            mask = rr_mask & np.isfinite(y)
            if np.count_nonzero(mask) < 2:
                continue

            x = rr_slice[mask]
            yy = y[mask]
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = yy[order]
            x_unique, unique_idx = np.unique(x_sorted, return_index=True)
            y_unique = y_sorted[unique_idx]
            if x_unique.size < 2:
                continue
            k = min(3, x_unique.size - 1)
            spline = UnivariateSpline(x_unique, y_unique, s=0, k=k)
            out[i_obs] = spline(rr_target)

        return out

    def _solve_rr_from_omega(self, omega_slice, rr_slice, omega_target):
        mask = np.isfinite(omega_slice) & np.isfinite(rr_slice) & (rr_slice >= rr_slice[0])
        if np.count_nonzero(mask) < 2:
            return np.nan

        rr = rr_slice[mask]
        omg = omega_slice[mask]
        order = np.argsort(rr)
        rr = rr[order]
        omg = omg[order]

        x0 = rr[np.argmin(np.abs(omg - omega_target))]

        rr_unique, unique_idx = np.unique(rr, return_index=True)
        omg_unique = omg[unique_idx]
        if rr_unique.size < 2:
            return np.nan
        k = min(3, rr_unique.size - 1)
        omg_rr_spline = UnivariateSpline(rr_unique, omg_unique, s=0, k=k)
        omg_rr_spline_der = omg_rr_spline.derivative()

        def f(rr_val):
            return omg_rr_spline(rr_val) - omega_target

        def fprime(rr_val):
            return omg_rr_spline_der(rr_val)

        try:
            rr_sol = newton(f, x0=x0, fprime=fprime, maxiter=50, tol=1e-10)
            if not np.isfinite(rr_sol):
                raise RuntimeError("non-finite Newton solution")
            rr_sol = float(np.clip(rr_sol, np.nanmin(rr), np.nanmax(rr)))
            return rr_sol
        except Exception:
            omg_order = np.argsort(omg)
            omg_s = omg[omg_order]
            rr_s = rr[omg_order]
            omg_unique, unique_idx = np.unique(omg_s, return_index=True)
            rr_unique = rr_s[unique_idx]

            if omg_unique.size >= 2 and np.all(np.diff(omg_unique) > 0):
                k_inv = min(3, omg_unique.size - 1)
                rr_omg_spline = UnivariateSpline(omg_unique, rr_unique, s=0, k=k_inv)
                return float(rr_omg_spline(omega_target))

            return float(rr[np.argmin(np.abs(omg - omega_target))])



    # for testing
    def recompute_kepler(self, spline_order=3):
        """
        Recompute r_ratio where Omega_p - Omega = 0 for each e_c index.

        This helper is intended for testing. It uses rotation states only
        (indices 1:11 in all_obs), where r_ratio spans [0.50, ..., 0.95].

        Returns
        -------
        np.ndarray
            Array of shape (n_ec,) containing the root r_ratio for each e_c.
            Entries are NaN where no zero-crossing is found.
        """
        all_obs = self._build_all_observables()

        # Rotation cases only: Omega_p (idx 13) - Omega (idx 3)
        Omega_p = all_obs[1:11, :, 13] 
        Omega   = all_obs[1:11, :, 3]
        delta = Omega_p - Omega
        n_ec = Omega_p.shape[1]
        rr_roots = np.full((n_ec,), np.nan, dtype=float)

        for i_ec in range(n_ec):
            rr = all_obs[1:11, i_ec, 14]
            y0 = Omega_p[:, i_ec]
            y1 = Omega[:, i_ec]
            dy = y0 - y1 

            mask = np.isfinite(rr) & np.isfinite(dy)
            if np.count_nonzero(mask) < 2:
                continue

            x  = rr[mask]
            y0 = y0[mask]
            y1 = y1[mask]
            dy = dy[mask]

            # Exact zero on sampled points.
            zero_idx = np.where(np.isclose(y0-y1, 0.0, atol=1e-12))[0]
            if zero_idx.size > 0:
                rr_roots[i_ec] = float(x[zero_idx[0]])
                continue

            k = min(spline_order, x.size - 1)
            if k < 1:
                continue

            spline = UnivariateSpline(x, dy, s=0, k=k)
            spline_der = spline.derivative()
            # spline0 = UnivariateSpline(x, y0, s=0, k=k)
            # spline0_der = spline.derivative()
            # spline1 = UnivariateSpline(x, y1, s=0, k=k)
            # spline1_der = spline.derivative()

            # Use Newton directly with initial guess at smallest |Omega_p - Omega|.
            x0 = float(x[np.argmin(np.abs(dy))])
            try:
                root = float(
                    newton(
                        lambda rr_val: spline(rr_val),
                        x0=x0,
                        fprime=lambda rr_val: spline_der(rr_val),
                        maxiter=50,
                        tol=1e-12,
                    )
                )
                # Constrain to the physically tabulated r_ratio range.
                rr_roots[i_ec] = root
            except Exception:
                rr_roots[i_ec] = np.nan

        # recompute observatives for kepler case
        obs_recomp_kep = np.full((n_ec, all_obs.shape[2]), np.nan, dtype=float)
        for i_obs, obs in enumerate(self.obs_names[:-1]):
            for i_ec in range(n_ec):
                spline_obs = UnivariateSpline(rr, all_obs[1:11, i_ec, i_obs], s=0, k=spline_order)
                obs_recomp_kep[i_ec, i_obs] = spline_obs(rr_roots[i_ec])
        
        obs_recomp_kep[:, 14] = rr_roots

        return obs_recomp_kep


rns_networks = RNSNetworks

__all__ = ["RNSNetworks", "rns_networks"]
