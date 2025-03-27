package student;

/**
 * This class is for creating the hand cards.
 *
 * @author Paul Liao
 */

public class Hand {

    /** The deck that game is using. */
    private Deck deck;
    /** The size of hand cards. */
    private int size;
    /** The cards in hand. */
    private Card[] hand;

    /**
     * Constructor that store the cards in the given size.
     *
     * @param d This is the cards will be in the hand.
     * @param s This is the size of the hand.
     */
    public Hand(Deck d, int s) {
        deck = d;
        size = s;
        hand = new Card[size];
        for (int i = 0; i < size; i++) {
            hand[i] = deck.draw();
        }
    }

    /**
     * It gets the size of hand.
     *
     * @return int The size of hand.
     */
    public int getSize() {
        return size;
    }

    /**
     * Get the card of given index.
     *
     * @param i This is the index of card.
     * @return Card The chosen card.
     */
    public Card get(int i) {
        if (i < 0 || i >= size) {
            System.out.println("Invalid hand index!");
            i = 0;
        }
        return hand[i];
    }

    /**
     * Remove the card and return true if success.
     *
     * @param card This is the card needs been removed.
     * @return boolean This returns true if the operation is success.
     */
    public boolean remove(Card card) {
        for (int i = 0; i < size; i++) {
            if (hand[i].equals(card)) {
                hand[i] = deck.draw();
                return true;
            }
        }
        return false;
    }

}
