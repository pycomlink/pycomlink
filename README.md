# Python tools for processing of commercial MW link data

## TODO
 * ~~~Function for saving Comlink object to HDF5~~~
     * Also write processing info to HDF5 file
 * ~~~Query metadata (location, frequency, etc.) from database and parse to Comlink object~~~
 * Define crucial metadata entries and their naming convention
 * Resolve protection links
 * Move over processing methods from lirage
     * ~~~wet/dry std_dev~~~
         * ~~~Asure that std_dev method wet/dry index is centered over window~~~
     * wet/dty SFTF
     * wet antenna Schleiss et al
     * ~~~A-R transformation~~~
     * Update docstrings!!!
 * Maybe refactor Comlink class
     * seperate near-far (_nf) and far-near (_fn) (plus other signals, protection, etc.)
       and apply processing individually. Then just define one link as the ensemble of
       its signals (_nf, _fn, _nf_protect, ...)
 * Is 'mhz_a' in metadata the frequency from near- to far-end or far to near?

