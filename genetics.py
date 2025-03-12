# genetics
import random
import numpy as np

class Gene:
    """Represents a genetic component with two alleles and dominance factors"""
    def __init__(self, allele1, allele2, dominance1=1, dominance2=0.5):
        """
        Initialize a gene with two alleles and their dominance values
        :param allele1: First allele value (float)
        :param allele2: Second allele value (float)
        :param dominance1: Dominance factor for first allele (0-1)
        :param dominance2: Dominance factor for second allele (0-1)
        """
        self.alleles = (allele1, allele2)
        self.dominance = (dominance1, dominance2)

    def express(self):
        """Return the expressed allele based on dominance hierarchy"""
        return self.alleles[0] if self.dominance[0] > self.dominance[1] else self.alleles[1]

    def mutate(self, mutation_rate=0.01, mutation_strength=0.1):
        """
        Apply random mutations to alleles and dominance factors
        :param mutation_rate: Probability of mutation (0-1)
        :param mutation_strength: Magnitude of mutation effect (0-1)
        """
        # Mutate allele values
        if random.random() < mutation_rate:
            self.alleles = (
                self.alleles[0] * random.uniform(1 - mutation_strength, 1 + mutation_strength),
                self.alleles[1] * random.uniform(1 - mutation_strength, 1 + mutation_strength)
            )

        # Mutate dominance factors with lower probability
        if random.random() < mutation_rate/2:
            self.dominance = (
                # First allele dominance mutation
                min(1, max(0, self.dominance[0] + random.uniform(-0.2, 0.2))),
                # Second allele dominance mutation
                min(1, max(0, self.dominance[1] + random.uniform(-0.2, 0.2)))
            )


class Genome:
    """Container class for all genetic information of an organism"""
    def __init__(self, parent1=None, parent2=None):
        """
        Initialize genome - either randomly or through sexual reproduction
        :param parent1: First parent genome (optional)
        :param parent2: Second parent genome (optional)
        """
        self.genes = {}

        if parent1 and parent2:
            self._crossover(parent1, parent2)  # Sexual reproduction
        else:
            self._generate_initial_genes()  # Asexual/initial generation

    def _generate_initial_genes(self):
        """Create randomized genes for initial population organisms"""
        '''self.genes.update({
            'new_trait': Gene(random.uniform(0, 1), random.uniform(0, 1))
        })'''
        self.genes = {
            # Strength: Impacts hunting efficiency and combat ability
            'strength': Gene(random.uniform(0.5, 0.6), random.uniform(0.5, 0.6)),
            # Speed: Movement capability (pixels per frame)
            'speed': Gene(random.uniform(0.8, 1.5), random.uniform(0.8, 1.5)),
            # Sight: Visual range (pixels)
            'sight_range': Gene(random.uniform(130, 140), random.uniform(130, 140)),
            # Sight: Visual field of view (pixels)
            'sight_fov': Gene(random.uniform(110, 120), random.uniform(110, 120)),
            # Size: Physical size and energy requirements
            'size': Gene(random.uniform(10, 11), random.uniform(10, 11)),
            # Lifespan: Maximum age in seconds
            'lifespan': Gene(random.uniform(80, 90), random.uniform(80, 90)),
            # Metabolism: Energy efficiency multiplier
            'metabolism': Gene(random.uniform(0.8, 1.0), random.uniform(0.8, 1.0))
        }

    def _crossover(self, parent1, parent2):
        """
        Combine genes from two parents using genetic crossover
        :param parent1: First parent genome
        :param parent2: Second parent genome
        """
        for trait in parent1.genes:
            # Random allele selection from each parent
            self.genes[trait] = Gene(
                random.choice(parent1.genes[trait].alleles),
                random.choice(parent2.genes[trait].alleles)
            )
            # Apply potential mutations to the new gene
            self.genes[trait].mutate()
    def get_color(self):
        """Calculate organism color based on genetic traits"""
        red = int(np.interp(self.get_trait('strength'), [0.0, 2], [0, 255]))
        green = int(np.interp(self.get_trait('speed'), [0.0, 1.5], [0, 255]))
        blue = int(np.interp(self.get_trait('sight_range'), [00, 160], [0, 255]))
        return (red, green, blue)
    def get_trait(self, trait_name):
        """
        Get the expressed value for a specific trait
        :param trait_name: Name of trait to retrieve
        :return: Expressed trait value
        """
        return self.genes[trait_name].express()

    # Pickling support for save/load functionality
    def __getstate__(self):
        """Prepare genome data for serialization"""
        return {trait: (gene.alleles, gene.dominance) for trait, gene in self.genes.items()}

    def __setstate__(self, state):
        """Reconstruct genome from serialized data with forward compatibility to allow genome expansion"""
        # Create base genome with current traits
        self._generate_initial_genes()
        
        # Update with saved values where they exist
        for trait, (alleles, dominance) in state.items():
            if trait in self.genes:
                self.genes[trait] = Gene(alleles[0], alleles[1], dominance[0], dominance[1])
        
        # Add any missing traits from current version
        current_traits = Genome().genes.keys()  # Get traits from fresh genome
        for trait in current_traits:
            if trait not in self.genes:
                # Generate new trait with default initial values
                self.genes[trait] = Gene(random.uniform(0.5, 2), random.uniform(0.5, 2))