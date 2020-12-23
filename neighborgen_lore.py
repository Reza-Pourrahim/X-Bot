import numpy as np

class LoreNeighborhoodGenerator(object):

    def __init__(self, blackbox, neigh_type, sample=1000, verbose=False):
        self.blackbox = blackbox
        self.neigh_type = neigh_type
        self.verbose = verbose
        self.sample = sample


    def vicinity_sampling(self, z, n=1000, threshold=None, kind="gaussian_matched", distribution=None,
                          distribution_kwargs=dict(), **kwargs):
        if self.verbose:
            print("\nSampling -->", kind)
        if kind == "gaussian":
            Z = self.gaussian_vicinity_sampling(z, threshold, n)
        elif kind == "gaussian_matched":
            Z = self.gaussian_matched_vicinity_sampling(z, threshold, n)
        elif kind == "gaussian_global":
            Z = self.gaussian_global_sampling(z, n)
        elif kind == "uniform_sphere":
            Z = self.uniform_sphere_vicinity_sampling(z, n, threshold)
        elif kind == "uniform_sphere_scaled":
            Z = self.uniform_sphere_scaled_vicinity_sampling(z, n, threshold)
        else:
            raise Exception("Vicinity sampling kind not valid")

        # qui aggiungere copia variabili non modificabili

        # qui aggiungere correzione categorici
        return Z


    def gaussian_matched_vicinity_sampling(self, z, epsilon, n=1):
        return self.gaussian_vicinity_sampling(z, epsilon, n) / np.sqrt(1 + (epsilon ** 2))


    def gaussian_vicinity_sampling(self, z, epsilon, n=1):
        return z + (np.random.normal(size=(n, z.shape[1])) * epsilon)


    def gaussian_global_sampling(self,z, n=1):
        return np.random.normal(size=(n, z.shape[1]))


    def uniform_sphere_origin(self, n, d, r=1):
        """Generate "num_points" random points in "dimension" that have uniform probability over the unit ball scaled
        by "radius" (length of points are in range [0, "radius"]).

        Parameters
        ----------
        n : int
            number of points to generate
        d : int
            dimensionality of each point
        r : float
            radius of the sphere

        Returns
        -------
        array of shape (n, d)
            sampled points
        """
        # First generate random directions by normalizing the length of a
        # vector of random-normal values (these distribute evenly on ball).
        random_directions = np.random.normal(size=(d, n))
        random_directions /= np.linalg.norm(random_directions, axis=0)
        # Second generate a random radius with probability proportional to
        # the surface area of a ball with a given radius.
        random_radii = np.random.random(n) ** (1 / d)
        # Return the list of random (direction & length) points.
        return r * (random_directions * random_radii).T


    def uniform_sphere_vicinity_sampling(self, z, n=1, r=1):
        Z = self.uniform_sphere_origin(n, z.shape[1], r)
        self.translate(Z, z)
        return Z


    def uniform_sphere_scaled_vicinity_sampling(self, z, n=1, threshold=1):
        Z = self.uniform_sphere_origin(n, z.shape[1], r=1)
        Z *= threshold
        self.translate(Z, z)
        return Z


    def translate(self, X, center):
        """Translates a origin centered array to a new center

        Parameters
        ----------
        X : array
            data to translate centered in the axis origin
        center : array
            new center point

        Returns
        -------
        None
        """
        for axis in range(center.shape[-1]):
            X[..., axis] += center[..., axis]


    def binary_sampling_search(self, z, z_label, blackbox, lower_threshold=0, upper_threshold=4, n=10000, n_batch=1000,
                               stopping_ratio=0.01, kind="gaussian_matched", vicinity_sampler_kwargs=dict(),
                               check_upper_threshold=True, final_counterfactual_search=True, downward_only=True,
                               **kwargs):
        if self.verbose:
            print("Binary sampling search:")

        # sanity check for the upper threshold
        if check_upper_threshold:
            for i in range(int(n / n_batch)):
                Z = self.vicinity_sampling(z=z, n=n_batch, threshold=upper_threshold, kind=kind,
                                           **vicinity_sampler_kwargs)
                y = blackbox.predict(Z)
                if not np.all(y == z_label):
                    break
            if i == list(range(int(n / n_batch)))[-1]:
                raise Exception("No counterfactual found, increase upper threshold or n_search.")

        change_lower = False
        latest_working_threshold = upper_threshold
        Z_counterfactuals = list()
        while lower_threshold / upper_threshold < stopping_ratio:
            if change_lower:
                if downward_only:
                    break
                lower_threshold = threshold
            threshold = (lower_threshold + upper_threshold) / 2
            change_lower = True
            if self.verbose:
                print("   Testing threshold value:", threshold)
            for i in range(int(n / n_batch)):
                Z = self.vicinity_sampling(z=z, n=n_batch, threshold=threshold, kind=kind, **vicinity_sampler_kwargs)
                y = blackbox.predict(Z)
                if not np.all(y == z_label):  # if we found already some counterfactuals
                    counterfactuals_idxs = np.argwhere(y != z_label).ravel()
                    Z_counterfactuals.append(Z[counterfactuals_idxs])
                    latest_working_threshold = threshold
                    upper_threshold = threshold
                    change_lower = False
                    break
        if self.verbose:
            print("   Best threshold found:", latest_working_threshold)
        if final_counterfactual_search:
            if self.verbose:
                print("   Final counterfactual search... (this could take a while)", end=" ")
            Z = self.vicinity_sampling(z=z, n=n, threshold=latest_working_threshold, kind=kind,
                                       **vicinity_sampler_kwargs)
            y = blackbox.predict(Z)
            counterfactuals_idxs = np.argwhere(y != z_label).ravel()
            Z_counterfactuals.append(Z[counterfactuals_idxs])
            if self.verbose:
                print("Done!")
        Z_counterfactuals = np.concatenate(Z_counterfactuals)
        closest_counterfactual = min(Z_counterfactuals, key=lambda p: sum((p - z.ravel()) ** 2))
        return closest_counterfactual.reshape(1, -1), latest_working_threshold

    def generate_fn(self, x, sample=1000, nbr_runs=10):
        y_val = self.blackbox.predict(x.reshape(1, -1))[0]

        if self.neigh_type == 'gmgm':
            cf, bt = self.binary_sampling_search(x.reshape(1, -1), y_val, self.blackbox, kind="gaussian_matched")
            Z = self.vicinity_sampling(z=cf, n=sample, threshold=bt, kind="gaussian_matched")
            Z = np.vstack([x.reshape(1, -1), cf, Z])

        elif self.neigh_type == 'gmus':
            cf, bt = self.binary_sampling_search(x.reshape(1, -1), y_val, self.blackbox, kind="gaussian_matched")
            Z = self.vicinity_sampling(z=cf, n=sample, threshold=bt, kind="uniform_sphere")
            Z = np.vstack([x.reshape(1, -1), cf, Z])

        elif self.neigh_type == 'gmgmx':
            cf, bt = self.binary_sampling_search(x.reshape(1, -1), y_val, self.blackbox, kind="gaussian_matched")
            Zcf = self.vicinity_sampling(z=cf, n=sample//2, threshold=bt, kind="gaussian_matched")
            Zx = self.vicinity_sampling(z=x.reshape(1, -1), n=sample//2, threshold=bt, kind="gaussian_matched")
            Z = np.vstack([x.reshape(1, -1), cf, Zcf, Zx])

        elif self.neigh_type == 'gmusx':
            cf, bt = self.binary_sampling_search(x.reshape(1, -1), y_val, self.blackbox, kind="gaussian_matched")
            Zcf = self.vicinity_sampling(z=cf, n=sample//2, threshold=bt, kind="uniform_sphere")
            Zx = self.vicinity_sampling(z=x.reshape(1, -1), n=sample//2, threshold=bt, kind="uniform_sphere")
            Z = np.vstack([x.reshape(1, -1), cf, Zcf, Zx])

        elif self.neigh_type == 'ngmusx':
            Z_list = list()
            for _ in range(nbr_runs - 1):
                cf, bt = self.binary_sampling_search(x.reshape(1, -1), y_val, self.blackbox, kind="gaussian_matched")
                Zcf = self.vicinity_sampling(z=cf, n=sample//nbr_runs, threshold=bt, kind="uniform_sphere")
                Zcf = np.vstack([cf, Zcf])
                Z_list.append(Zcf)

            Zx = self.vicinity_sampling(z=x.reshape(1, -1), n=sample//nbr_runs, threshold=bt, kind="uniform_sphere")
            Z = np.vstack([x.reshape(1, -1), Zx, np.vstack(Z_list)])

        else:
            print('unknown neighborhood generator')
            raise Exception

        return Z
