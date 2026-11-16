Adapting Class: hold the target model, adapt it and return its prediction

Augmentation Interface, with different Augmentation Strategies

Region Proposal Interface, with different Proposal Strategies

Refined Region Fusion Interface, with different Fusion strategies

Learning Interface, with different learning strategy (for now use the original redifm formula). It will own everything after the part you create 2 supervision.

Model Interface with different models

a DataLoader class, and subclass of DataLoader will dictate the order of test image.

An evaluate_non_adapting class
An evaluate_adapting class orchestrate subclass of DataLoader and Adapting to score.

---------
Move everything outside crop_teach to an archive folder except for the README.md and two
Make a seperate and independent refactor_crop_teach folder
For now no need to worry about Training
Forget about DinoV3 completely
Every class that implement an interface must have unit test
Remember to test the refactor by comparing some output of before and after refactor.
