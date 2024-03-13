def split_vector(self, V, Marker, SecLength):
        """
        Splits a vector into multiple sections based on marker positions.

        Parameters:
        V (numpy.ndarray): The input vector.
        Marker (list): List of marker positions.
        SecLength (int): Length of each section.

        Returns:
        numpy.ndarray: A 2D array containing the split sections of the vector.
        """
        SecLength = int(SecLength)
        M = np.zeros((len(Marker), SecLength))
        for i, marker in enumerate(Marker):
            marker = int(marker)
            epoch = V[marker:(marker + SecLength)]
            M[i, :len(epoch)] = epoch
        return M