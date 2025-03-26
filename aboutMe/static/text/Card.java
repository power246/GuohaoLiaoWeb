package student;

/**
 * This class is for creating the cards.
 *
 * @author Paul Liao
 */

public class Card {

    /** The rank of the card. */
    private int rank;
    /** The suit of the card. */
    private int suit;
    /** Denotes if it is a valid card. */
    private boolean flag = false;

    /**
     * Constructor that create a new card.
     *
     * @param r This is the rank of cards.
     * @param s This is the suit of cards.
     */
    public Card(int r, int s) {
        rank = r;
        suit = s;
        if (r < 1 || r > 13 || s < 1 || s > 4) {
            flag = true;
            System.out.println("Invalid student.Card");
        }
    }

    /**
     * It gets the int number of rank.
     *
     * @return int This returns the int number of rank.
     */
    public int getRankNum() {
        return rank;
    }

    /**
     * It gets the int number of suit.
     *
     * @return int This returns the int number of suit.
     */
    public int getSuitNum() {
        return suit;
    }

    /**
     * It gets the string name of rank.
     *
     * @return String This returns the string name of rank.
     */
    public String getRankName() {
        String name = null;
        if (rank == 1) {
            name = "Ace";
        } else if (rank == 2) {
            name = "Two";
        } else if (rank == 3) {
            name = "Three";
        } else if (rank == 4) {
            name = "Four";
        } else if (rank == 5) {
            name = "Five";
        } else if (rank == 6) {
            name = "Six";
        } else if (rank == 7) {
            name = "Seven";
        } else if (rank == 8) {
            name = "Eight";
        } else if (rank == 9) {
            name = "Nine";
        } else if (rank == 10) {
            name = "Ten";
        } else if (rank == 11) {
            name = "Jack";
        } else if (rank == 12) {
            name = "Queen";
        } else if (rank == 13) {
            name = "King";
        }
        if (flag) {
            name = "Ace";
        }
        return name;
    }

    /**
     * It gets the string name of the cade.
     *
     * @return String This returns the string name of the cade.
     */
    public String getSuitName() {
        String name = null;
        if (suit == 1) {
            name = "Spades";
        } else if (suit == 2) {
            name = "Hearts";
        } else if (suit == 3) {
            name = "Clubs";
        } else if (suit == 4) {
            name = "Diamonds";
        }
        if (flag) {
            name = "Spades";
        }
        return name;
    }

    /**
     * It overrides the toString method to get the readable name of the card.
     *
     * @return String This returns the readable name of the card.
     */
    @Override
    public String toString() {
        // Change the output of print statement.
        return getRankName() + " of " + getSuitName();
    }

    /**
     * It overrides the equals method to compare if the card is the same.
     *
     * @param obj This is the other card been compared.
     * @return boolean This returns true only if the rank and suit is the same.
     */
    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Card)) {
            return false;
        }
        if (obj == null) {
            return false;
        }
        Card other = (Card) obj;
        if (rank == other.getRankNum() && suit == other.getSuitNum()) {
            return true;
        }
        return false;
    }
}
