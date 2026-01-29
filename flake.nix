{
  description = "XeLaTeX project with amssymb, enumitem, Devanagari + Chinese";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
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
            xecjk         # nicer Chinese handling on XeTeX
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
            biblatex
            prettyref
            subfigure
            forest
            tasks
            fmtcount
            csquotes
            tikzlings
            tikzducks
            hyperref;
        };

        # Fonts for Unicode scripts (system fonts for XeTeX via fontconfig)
        fonts = [
          pkgs.noto-fonts            # Latin + many scripts
#          pkgs.noto-fonts-cjk        # Chinese/Japanese/Korean
          pkgs.noto-fonts-emoji
          pkgs.noto-fonts-extra      # includes Noto Serif/Sans Devanagari
#          pkgs.noto-fonts-cjk-serif
          pkgs.noto-fonts-lgc-plus
          pkgs.libertinus
          pkgs.dejavu_fonts
        ];

        fontsConf = pkgs.makeFontsConf { fontDirectories = fonts; };
        dependencies = [ tex pkgs.ghostscript pkgs.inkscape pkgs.caddy (pkgs.python3Full.withPackages (p:  [ p.pyopengl p.pyglm p.glfw p.pyx p.sounddevice p.mido p.numba p.matplotlib ])) ] ++ fonts;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = dependencies;
          # Make fonts visible to XeTeX in the shell
          FONTCONFIG_FILE = fontsConf;
        };
        devShells.texOnly = pkgs.mkShell {
          packages = [ tex pkgs.ghostscript ];
          # Make fonts visible to XeTeX in the shell
          FONTCONFIG_FILE = fontsConf;
        };
      });
}
