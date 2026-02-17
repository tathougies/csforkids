{
  description = "XeLaTeX project with amssymb, enumitem, Devanagari + Chinese";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # TeX (just what we need)
        tex = pkgs.texlive.combine {
          inherit (pkgs.texlive)
            scheme-small
            xetex         # XeLaTeX engine
            latexmk
            fontspec
            polyglossia
            qrcode
            eso-pic
            zref
            xecjk         # nicer Chinese handling on XeTeX
            needspace
            newtx
            amsfonts amsmath # amssymb
            enumitem
            gfsdidot
            greek-fontenc
            cbfonts-fd
            pxfonts
            mlmodern
            accents
            tipa
            tcolorbox
            environ
            tikzfill
            tikzmark
            tikzpagenodes
            ifoddpage
            pdfcol
            titlesec
            listings
            fancyhdr
            tufte-latex
            hardwrap
            catchfile
            biber
            capt-of
            biblatex
            prettyref
            subfigure
            subfig
            forest
            tasks
            fmtcount
            csquotes
            tikzlings
            tikzducks
            newfloat
            flowchart
            makeshape
            hyperref;
        };

        # Fonts for Unicode scripts (system fonts for XeTeX via fontconfig)
        fonts = [
          pkgs.noto-fonts            # Latin + many scripts
#          pkgs.noto-fonts-cjk        # Chinese/Japanese/Korean
          pkgs.noto-fonts-color-emoji
#          pkgs.noto-fonts-extra      # includes Noto Serif/Sans Devanagari
#          pkgs.noto-fonts-cjk-serif
          pkgs.noto-fonts-lgc-plus
          pkgs.libertinus
          pkgs.dejavu_fonts
        ];

        fontsConf = pkgs.makeFontsConf { fontDirectories = fonts; };
        mkdocsPkgs = p: [p.mkdocs p.mkdocs-gen-files p.mkdocs-literate-nav (p.callPackage ./nix/mkdocs-bootswatch.nix {}) p.jinja2 (p.callPackage ./nix/jinja2-simple-tags.nix {}) ];
        dependencies = [ pkgs.ispell tex pkgs.imagemagick pkgs.ghostscript pkgs.inkscape pkgs.caddy pkgs.ninja pkgs.nodejs pkgs.py-spy
                         (pkgs.python3.withPackages (p: mkdocsPkgs p ++ [
                                                           p.tkinter p.pyopengl
                                                           p.tkinter-gl p.pyglm p.pillow p.glfw  p.pygame p.pyx p.sounddevice
                                                           p.mido p.numba p.matplotlib ])) ] ++ fonts;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = dependencies;
          # Make fonts visible to XeTeX in the shell
          FONTCONFIG_FILE = fontsConf;
        };
        devShells.texOnly = pkgs.mkShell {
          packages = [ tex pkgs.ghostscript pkgs.nodejs (pkgs.python3.withPackages mkdocsPkgs) ];
          # Make fonts visible to XeTeX in the shell
          FONTCONFIG_FILE = fontsConf;
        };
      });
}
