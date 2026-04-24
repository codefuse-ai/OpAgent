"""Internal-only online webagent helper was removed from the open-source release.

This repository does not include credentials, private endpoints, or internal service
integrations required by the original script. If you need similar functionality,
create a local/private script and inject configuration through environment variables.
"""

raise RuntimeError(
    "This script is not included in the open-source release because it depended on "
    "private services and credentials."
)
