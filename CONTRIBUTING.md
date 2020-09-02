# Contribute to the NVIDIA Model Analyzer

Want to hack on the NVIDIA Model Analyzer Project? Awesome!
We only require you to sign your work, the below section describes this!
You must also agree to the [CLA] and follow the rules below.

## Sign your work

The sign-off is a simple line at the end of the explanation for the patch. Your
signature certifies that you wrote the patch or otherwise have the right to pass
it on as an open-source patch. The rules are pretty simple: if you can certify
the below (from [developercertificate.org](http://developercertificate.org/)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Then, you just add a line to every git commit message:

    Signed-off-by: Joe Smith <joe.smith@email.com>

Use your real name. Sorry, no pseudonyms or anonymous contributions.

If you set your `user.name` and `user.email` git configs, you can sign your
commit automatically with `git commit -s`.

## Contribution Rules

- In general, please follow the existing conventions in the
  relevant file, submodule, module, and project when you add new code
  or when you extend/fix existing functionality.

- Avoid introducing unnecessary complexity into existing code so that
  maintainability and readability are preserved.

- Try to keep pull requests (PRs) as concise as possible:

  - Avoid committing commented-out code.

  - Wherever possible, each PR should address a single concern. If
    there are several otherwise-unrelated things that should be fixed
    to reach a desired endpoint, it is perfectly fine to open several
    PRs and state in the description which PR depends on another
    PR. The more complex the changes are in a single PR, the more time
    it will take to review those changes.

  - Make sure that the build log is clean, meaning no warnings or
    errors should be present.

- Make sure all tests in the `tests` directory pass:

- Make sure that you can contribute your work to open source (no
  license and/or patent conflict is introduced by your code).

- Thanks in advance for your patience as we review your contributions;
  we do appreciate them!

[CLA]: CLA.md