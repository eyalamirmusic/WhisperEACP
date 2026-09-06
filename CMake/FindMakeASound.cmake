include(CPM)

# Fetched after eacp, which brings Miro and ea_data_structures along — CPM keys
# on the package name, so MakeASound finds both already populated rather than
# adding a second copy of either under a different tag.
CPMAddPackage(
        NAME MakeASound
        GITHUB_REPOSITORY eyalamirmusic/makeasound
        GIT_TAG main)
