{
  description = "Python environment for XML Zuschreibung analysis and embedding evaluation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Python environment with all required packages
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          # Core dependencies
          sentence-transformers
          numpy
          scipy
          matplotlib
          pandas
          lxml
          nltk
          scikit-learn
          openai
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [ pythonEnv ];

          shellHook = ''
            echo "✅ Python environment ready for XML analysis, embeddings, and knowledge extraction"
            echo "   Model: paraphrase-multilingual-MiniLM-L12-v2"
            echo "   Run: python 2026-analyses-summary/extract_opac_knowledge.py"
            python --version
          '';
        };

        packages.default = pythonEnv;
      }
    );
}
